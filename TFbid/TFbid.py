# bridge_bid_transformer.py
import json, math, random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_linear_schedule_with_warmup, LogitsProcessor

# ---------------------------
# 1) 体系化されたビッド集合
# ---------------------------
BIDS_ORDERED = [
    "1C","1D","1H","1S","1NT",
    "2C","2D","2H","2S","2NT",
    "3C","3D","3H","3S","3NT",
    "4C","4D","4H","4S","4NT",
    "5C","5D","5H","5S","5NT",
    "6C","6D","6H","6S","6NT",
    "7C","7D","7H","7S","7NT",
]
CALLS = ["P"] + BIDS_ORDERED + ["X","XX"]  # 1 + 35 + 2 = 38

# 比較用のスート優先度（C < D < H < S < NT）
SUIT_ORDER = {"C":0,"D":1,"H":2,"S":3,"NT":4}
def bid_rank(b: str) -> Optional[tuple]:
    """ '2H' -> (2, 'H'), 'P','X','XX' -> None """
    if b in ("P","X","XX"):
        return None
    if b.endswith("NT"):
        level = int(b[:-2]); suit = "NT"
    else:
        level = int(b[:-1]); suit = b[-1]
    return (level, suit)

def higher_than(a: str, b: str) -> bool:
    """ a は b より高い入札か？ """
    ra, rb = bid_rank(a), bid_rank(b)
    if ra is None or rb is None:
        return False
    la, sa = ra; lb, sb = rb
    return (la > lb) or (la == lb and SUIT_ORDER[sa] > SUIT_ORDER[sb])

def current_contract(auction: List[str]) -> Optional[str]:
    """ 直近の最高入札（ビッド）を返す。無ければ None """
    top = None
    for c in auction:
        if bid_rank(c) is not None:
            if top is None or higher_than(c, top):
                top = c
    return top

def side_to_act(auction: List[str], dealer: str) -> str:
    """ 今までのコール数から手番の座席を返す """
    order = ["N","E","S","W"]
    # ディーラーを0に回す
    shift = {"N":0,"E":1,"S":2,"W":3}[dealer]
    rotated = order[shift:] + order[:shift]
    idx = len(auction) % 4
    return rotated[idx]

def last_call_by_side(auction: List[str], side: str, dealer: str) -> Optional[str]:
    """ side('NS' or 'EW') の直近コール。 """
    seats = {
        "NS": {"N","S"},
        "EW": {"E","W"}
    }
    order = ["N","E","S","W"]
    shift = {"N":0,"E":1,"S":2,"W":3}[dealer]
    rotated = order[shift:] + order[:shift]
    last = None
    for i,c in enumerate(auction):
        who = rotated[i % 4]
        if who in seats[side]:
            last = c
    return last

def last_call(auction: List[str]) -> Optional[str]:
    return auction[-1] if auction else None

def is_our_side(seat: str, who: str) -> bool:
    return (seat in ("N","S") and who in ("N","S")) or (seat in ("E","W") and who in ("E","W"))

# ---------------------------
# 2) 合法性マスク
# ---------------------------
def legal_mask(auction: List[str], dealer: str, seat: str) -> List[bool]:
    """ CALLS と同じ順序で合法/違法のマスクを返す """
    mask = [False]*len(CALLS)
    top = current_contract(auction)
    # Pass は常に合法（オークション未終了前）
    mask[CALLS.index("P")] = True

    # ビッド合法性
    for b in BIDS_ORDERED:
        if top is None or higher_than(b, top):
            mask[CALLS.index(b)] = True

    # X/XX
    lc = last_call(auction)
    if lc is not None and bid_rank(lc) is not None:
        # 直前がビッド → Xの可能性
        # ただし、直前が自サイドのビッドならX不可
        # 手番を求めて直前の席を算出
        # dealer回りを使って席列を作る
        order = ["N","E","S","W"]
        shift = {"N":0,"E":1,"S":2,"W":3}[dealer]
        rotated = order[shift:] + order[:shift]
        prev_who = rotated[(len(auction)-1) % 4]
        if not is_our_side(seat, prev_who):
            mask[CALLS.index("X")] = True

    # XXは自サイドがダブルされているときのみ
    # ＝直近の「X」が相手サイドから自サイドのコールに対して出ている状態
    if "X" in auction:
        # 直近のXの位置と誰が出したかを見て、自サイドがダブルされているならXX可
        order = ["N","E","S","W"]
        shift = {"N":0,"E":1,"S":2,"W":3}[dealer]
        rotated = order[shift:] + order[:shift]
        for i in range(len(auction)-1, -1, -1):
            if auction[i] == "X":
                who = rotated[i % 4]
                # そのXは我々に対するXだった（＝Xを出したのは相手サイド）
                if not is_our_side(seat, who):
                    mask[CALLS.index("XX")] = True
                break

    return mask

class BridgeLegalityProcessor(LogitsProcessor):
    def __init__(self, tokenizer, dealer, seat, auction_prefix_ids, call_to_id):
        super().__init__()
        self.tokenizer = tokenizer
        self.dealer = dealer
        self.seat = seat
        self.auction_prefix_ids = auction_prefix_ids  # 生成の前半に含めた AUCTION のテキストID
        self.call_to_id = call_to_id

    def _extract_auction_so_far(self, input_ids: torch.LongTensor) -> List[str]:
        # 入力列から "AUCTION | " 以降～生成位置直前までの文字列を復元し、最後の空白区切りでコール列に
        text = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
        if "AUCTION" not in text:
            return []
        t = text.split("AUCTION",1)[1]
        # 例: " | P 1H P 2H \nTO_MOVE:N\n" のような形を想定
        t = t.split("\n")[0]
        if "|" in t:
            t = t.split("|",1)[1]
        calls = [c for c in t.strip().split() if c in CALLS]
        return calls

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        auction = self._extract_auction_so_far(input_ids)
        mask = legal_mask(auction, self.dealer, self.seat)
        # CALLS→語彙ID の集合にマップして不許可を -inf
        vocab_mask = torch.zeros_like(scores)
        # まず全て -inf
        vocab_mask[:] = float("-inf")
        # 許可コールのみスコアをそのまま
        for ok, call in zip(mask, CALLS):
            if ok:
                tid = self.call_to_id[call]
                vocab_mask[0, tid] = scores[0, tid]
        return vocab_mask

# ---------------------------
# 3) トークナイザ（手札→テキスト）
# ---------------------------
RANKS = "AKQJT98765432"
def encode_hand(hand: Dict[str,str]) -> str:
    def norm(suit):  # 未記載スートにも対応
        v = hand.get(suit, "")
        # ランクの順番を保証
        return "".join([r for r in RANKS if r in v])
    return f"S:{norm('S')} H:{norm('H')} D:{norm('D')} C:{norm('C')}"

def build_prompt(ex: Dict[str,Any]) -> str:
    # dealer: 'N','E','S','W'
    # vul: 'None','NS','EW','Both' など自由設計（ここでは例示的に 'NS','EW','Both','None'）
    # seat: 'N','E','S','W'
    header = f"<START> D:{ex['dealer']} V:{ex['vul']} Seat:{ex['seat']}\n"
    hand = f"HAND {encode_hand(ex['hand'])}\n"
    auction = "AUCTION | " + " ".join(ex["auction"]) + "\n"
    to_move = f"TO_MOVE:{ex['seat']}\n"
    return header + hand + auction + to_move

# ---------------------------
# 4) データセット
# ---------------------------
class AuctionDataset(Dataset):
    def __init__(self, path_jsonl: str, tokenizer, calls_vocab: List[str]):
        self.data = []
        with open(path_jsonl, "r") as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tok = tokenizer
        self.calls = calls_vocab

        # “出力ラベル”はテキスト末尾に 1 トークンとして付ける
        # 例: ... + "ANSWER " + "2H"
    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        ex = self.data[i]
        prompt = build_prompt(ex)
        answer = ex["label"]  # 例: "2H"
        text = prompt + "ANSWER " + answer
        ids = self.tok(text, return_tensors="pt")
        input_ids = ids["input_ids"][0]
        attn = ids["attention_mask"][0]
        # 教師ありのため “最後の1トークン（answer）” のみ損失対象にする
        labels = torch.full_like(input_ids, -100)
        labels[-1] = input_ids[-1]
        return input_ids, attn, labels, ex

def collate(batch):
    input_ids, attn, labels, exs = zip(*batch)
    maxlen = max(x.size(0) for x in input_ids)
    def pad(x, pad_id):
        out = torch.full((len(x), maxlen), pad_id, dtype=torch.long)
        for i,t in enumerate(x):
            out[i, :t.size(0)] = t
        return out
    pad_id = 0
    return {
        "input_ids": pad(input_ids, pad_id),
        "attention_mask": pad(attn, 0),
        "labels": pad(labels, -100),
        "examples": exs
    }

# ---------------------------
# 5) 学習
# ---------------------------
def train(model_name: str, train_path: str, out_dir: str, epochs=3, bs=16, lr=5e-5, warmup_ratio=0.06):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = AuctionDataset(train_path, tok, CALLS)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collate)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tok))
    model.train()

    steps = len(dl)*epochs
    opt = AdamW(model.parameters(), lr=lr)
    sch = get_linear_schedule_with_warmup(opt, int(steps*warmup_ratio), steps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for ep in range(epochs):
        total = 0.0
        for batch in dl:
            opt.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device)
            )
            loss = out.loss
            loss.backward()
            opt.step(); sch.step()
            total += loss.item()
        print(f"epoch {ep+1}: loss={total/len(dl):.4f}")

    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print("saved to", out_dir)

# ---------------------------
# 6) 推論（合法ビッドだけ）
# ---------------------------
@torch.no_grad()
def predict_next_call(model_dir: str, example: Dict[str,Any]) -> str:
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    prompt = build_prompt(example) + "ANSWER "
    inputs = tok(prompt, return_tensors="pt").to(device)

    # CALLトークンの語彙ID辞書
    call_to_id = {c: tok.convert_tokens_to_ids(tok.tokenize(c))[0] if len(tok.tokenize(c))==1
                  else tok.encode(c, add_special_tokens=False)[0]
                  for c in CALLS}

    # ロジットプロセッサで合法のみ許す
    lp = BridgeLegalityProcessor(
        tokenizer=tok,
        dealer=example["dealer"],
        seat=example["seat"],
        auction_prefix_ids=None,
        call_to_id=call_to_id
    )

    out = model.generate(
        **inputs,
        max_new_tokens=1,          # 次の1コールだけ
        do_sample=False,
        logits_processor=[lp]
    )
    gen = tok.decode(out[0], skip_special_tokens=True)
    # 最後のトークンが答え
    pred = gen.strip().split()[-1]
    if pred not in CALLS:
        # 念のためフォールバック：最高確率の合法から手動選択
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]
            auction = example["auction"]
            mask = legal_mask(auction, example["dealer"], example["seat"])
            scores = logits[0].clone()
            # トークンID→CALL の逆引き
            id_to_call = {v:k for k,v in call_to_id.items()}
            for tid in range(scores.size(0)):
                c = id_to_call.get(tid)
                if c is None or not mask[CALLS.index(c)]:
                    scores[tid] = -1e9
            pred_id = int(torch.argmax(scores).item())
            pred = id_to_call[pred_id]
    return pred

# ---------------------------
# 7) 使い方サンプル
# ---------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train","predict"], required=True)
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--train_path")
    ap.add_argument("--out_dir", default="./bridge-bid-model")
    ap.add_argument("--example_json")
    args = ap.parse_args()

    if args.mode == "train":
        assert args.train_path is not None
        train(args.model, args.train_path, args.out_dir)
    else:
        ex = json.loads(open(args.example_json).read())
        print(predict_next_call(args.out_dir, ex))
