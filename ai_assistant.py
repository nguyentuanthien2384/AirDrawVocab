"""
AirDrawVocab - AI Assistant Module (100% OFFLINE)
4 tinh nang AI: Chatbot, Smart Difficulty, Ensemble, Examples
Khong can API key, internet, thu vien them.
"""
import time, random
import numpy as np

WORD_DATABASE = {
    "apple": {"explain": "Apple - Qua tao. Meo: Apple bat dau bang A! 'An apple a day keeps the doctor away'",
        "hint": "1) Ve hinh tron to  2) Them cuong nho phia tren  3) Them 1 la ben canh cuong",
        "examples": [("I eat an apple every morning.","Toi an mot qua tao moi sang."),("The apple is red and sweet.","Qua tao do va ngot."),("Apple juice is my favorite drink.","Nuoc ep tao la do uong yeu thich cua toi.")]},
    "baseball": {"explain": "Baseball - Bong chay. BASE+BALL. Mon the thao quoc dan cua My va Nhat!",
        "hint": "1) Ve hinh tron to  2) Ve 2 duong cong chu S ben trong  3) Them net khau",
        "examples": [("He plays baseball after school.","Anh ay choi bong chay sau gio hoc."),("Baseball is very popular in Japan.","Bong chay rat pho bien o Nhat.")]},
    "book": {"explain": "Book - Quyen sach. 'Don't judge a book by its cover' - Dung danh gia qua ve ngoai.",
        "hint": "1) Ve hinh chu nhat dung nghieng  2) Them gay sach ben trai  3) Ve 2-3 duong ngang",
        "examples": [("She reads a book before bed.","Co ay doc sach truoc khi ngu."),("I borrowed a book from the library.","Toi muon sach tu thu vien.")]},
    "bowtie": {"explain": "Bowtie - No that co. BOW(no)+TIE(ca vat). Thuong thay o boi ban, MC!",
        "hint": "1) Ve 2 tam giac doi xung, dinh cham nhau  2) Them nut tron nho o giua",
        "examples": [("He wore a red bowtie to the party.","Anh ay deo no do den bua tiec."),("A bowtie looks very elegant.","No that co trong rat lich lam.")]},
    "diamond": {"explain": "Diamond - Kim cuong/Hinh thoi. 'Diamonds are forever' - Kim cuong la vinh cuu!",
        "hint": "1) Ve hinh thoi - 4 canh bang nhau  2) Nhon tren va duoi, rong 2 ben",
        "examples": [("The diamond ring sparkles.","Chiec nhan kim cuong lap lanh."),("A diamond has four equal sides.","Hinh thoi co bon canh bang nhau.")]},
    "dog": {"explain": "Dog - Con cho. 'A dog is man's best friend'. 'Hot dog' = xuc xich!",
        "hint": "1) Ve dau tron + than oval  2) Them 4 chan  3) Them tai, duoi, mat, mui",
        "examples": [("My dog likes to play in the park.","Con cho toi thich choi trong cong vien."),("The dog is barking loudly.","Con cho dang sua to.")]},
    "door": {"explain": "Door - Canh cua. 'When one door closes, another opens'.",
        "hint": "1) Ve hinh chu nhat dung  2) Them tay nam tron ben phai  3) Them ban le ben trai",
        "examples": [("Please close the door.","Xin hay dong cua."),("Someone is knocking on the door.","Ai do dang go cua.")]},
    "envelope": {"explain": "Envelope - Phong bi. EN(vao)+VELOPE(boc). Icon email van la hinh phong bi!",
        "hint": "1) Ve hinh chu nhat ngang  2) Ve chu V lon nguoc tu 2 goc tren  3) Dinh V cham giua",
        "examples": [("Put the letter in the envelope.","Bo la thu vao phong bi."),("I need to buy some envelopes.","Toi can mua vai cai phong bi.")]},
    "eye": {"explain": "Eye - Mat. Phat am giong 'ai'! 'Keep an eye on' = trong chung.",
        "hint": "1) Ve hinh oval nam ngang  2) Ve vong tron nho ben trong  3) Them cham den (dong tu)",
        "examples": [("She has beautiful blue eyes.","Co ay co doi mat xanh dep."),("Close your eyes and make a wish.","Nham mat lai va uoc mot dieu.")]},
    "fish": {"explain": "Fish - Con ca. So nhieu van la 'fish'! 'Like a fish out of water' = boi roi.",
        "hint": "1) Ve than hinh oval  2) Them duoi tam giac phia sau  3) Them mat tron va vay",
        "examples": [("The fish swims in the river.","Con ca boi trong song."),("We had fish for dinner.","Chung toi an ca cho bua toi.")]},
    "hat": {"explain": "Hat - Cai mu. 'Hats off to you!' = Xin nguong mo ban!",
        "hint": "1) Ve ban nguyet (dinh mu)  2) Them duong ngang rong hon (vanh mu)",
        "examples": [("Wear a hat to protect from the sun.","Doi mu de chong nang."),("The magician pulled a rabbit from his hat.","Nha ao thuat rut tho tu mu.")]},
    "leaf": {"explain": "Leaf - Chiec la. So nhieu 'leaves'. 'Turn over a new leaf' = Lam lai cuoc doi.",
        "hint": "1) Ve hinh oval nhon hai dau  2) Them gan la o giua  3) Them cuong ngan phia duoi",
        "examples": [("The leaf fell from the tree.","Chiec la roi tu tren cay."),("Leaves change color in autumn.","La doi mau vao mua thu.")]},
    "lightning": {"explain": "Lightning - Tia set. LIGHT+NING = anh sang cuc nhanh! 'Lightning fast'!",
        "hint": "1) Ve duong zigzag gap khuc tu tren xuong  2) 2-3 lan gap khuc nhon",
        "examples": [("Lightning struck the old tree.","Set danh trung cay co thu."),("He runs as fast as lightning.","Anh ay chay nhanh nhu chop.")]},
    "moon": {"explain": "Moon - Mat trang. 'Once in a blue moon' = Rat hiem khi. 'Over the moon' = vui mung.",
        "hint": "1) Ve hinh luoi liem  2) Net cong lon ben trai + net cong nho ben phai",
        "examples": [("The moon is bright tonight.","Mat trang sang toi nay."),("The full moon looks beautiful.","Trang tron trong that dep.")]},
    "pants": {"explain": "Pants - Quan dai. Luon so nhieu (2 ong)! O Anh 'pants' = quan lot.",
        "hint": "1) Ve hinh thang nguoc (that lung)  2) Them 2 ong quan phia duoi",
        "examples": [("He bought new pants for work.","Anh ay mua quan moi de di lam."),("These pants are too long.","Chiec quan nay dai qua.")]},
    "scissors": {"explain": "Scissors - Cai keo. Luon so nhieu! 'A pair of scissors'. Chu C cam!",
        "hint": "1) Ve hinh chu X (2 luoi cheo)  2) Them 2 vong tron nho (tay cam)  3) Phan tren nhon dan",
        "examples": [("Use scissors to cut the paper.","Dung keo de cat giay."),("Rock, paper, scissors is fun.","Bua, bao, keo la tro choi vui.")]},
    "square": {"explain": "Square - Hinh vuong. Times Square o New York! 'Fair and square' = cong bang.",
        "hint": "1) Ve 4 canh bang nhau  2) 4 goc vuong  3) Don gian nhat!",
        "examples": [("Draw a square on the paper.","Ve hinh vuong tren giay."),("A square has four equal sides.","Hinh vuong co bon canh bang nhau.")]},
    "star": {"explain": "Star - Ngoi sao. 'Reach for the stars' = Vuon toi nhung vi sao!",
        "hint": "1) Ve 5 dinh nhon  2) Noi cac dinh bang net lien zigzag",
        "examples": [("The stars shine at night.","Nhung ngoi sao lap lanh ban dem."),("She wants to be a movie star.","Co ay muon tro thanh ngoi sao dien anh.")]},
    "t-shirt": {"explain": "T-shirt - Ao phong. Goi la T-shirt vi trai phang co hinh chu T!",
        "hint": "1) Ve than ao hinh thang  2) Them 2 tay ao ngan 2 ben  3) Them co ao V/U",
        "examples": [("I wear a t-shirt in summer.","Toi mac ao phong vao mua he."),("This t-shirt is made of cotton.","Ao phong nay lam tu cotton.")]},
}

class SmartDifficulty:
    def __init__(self):
        self.history = []
        self.word_difficulty = {}
        self.player_skill = 0.5
        self.base_difficulty = {"square":0.1,"star":0.2,"moon":0.2,"apple":0.3,"book":0.3,"door":0.3,
            "eye":0.3,"hat":0.3,"leaf":0.3,"fish":0.4,"diamond":0.4,"envelope":0.4,
            "lightning":0.4,"baseball":0.5,"pants":0.5,"dog":0.6,"bowtie":0.6,"t-shirt":0.6,"scissors":0.7}

    def record_result(self, word, correct, time_taken, confidence):
        self.history.append({"word":word,"correct":correct,"time":time_taken,"confidence":confidence})
        if word not in self.word_difficulty:
            self.word_difficulty[word] = self.base_difficulty.get(word, 0.5)
        self.word_difficulty[word] = max(0.1, self.word_difficulty[word]-0.1) if correct else min(1.0, self.word_difficulty[word]+0.15)
        recent = self.history[-10:]
        if recent:
            wr = sum(1 for r in recent if r["correct"])/len(recent)
            at = np.mean([r["time"] for r in recent])
            self.player_skill = wr*0.7 + max(0,(60-at)/60)*0.3

    def sort_words(self, word_list):
        return sorted(word_list, key=lambda w: abs(self.word_difficulty.get(w, self.base_difficulty.get(w,0.5)) - self.player_skill))

    def should_hint(self, word, elapsed, has_content):
        if not has_content and elapsed > 25: return True, "draw"
        if has_content and elapsed > 40: return True, "retry"
        return False, ""

    def get_encouragement(self, correct, streak):
        if correct:
            if streak >= 5: return "XUAT SAC! Chuoi {} lien tiep!".format(streak)
            if streak >= 3: return "Tuyet voi! Combo {}!".format(streak)
            return "Chinh xac! Tiep tuc nhe!"
        return random.choice(["Thu lai nhe! Ve ro net hon!","Meo: Ve to va ro rang!","Nhan H de xem goi y!"])

    def get_difficulty_label(self):
        if self.player_skill < 0.3: return "DE", (100,200,100)
        if self.player_skill < 0.6: return "TRUNG BINH", (200,200,100)
        return "KHO", (200,100,100)

class EnsemblePredictor:
    def __init__(self, base_model):
        self.model = base_model
        self.vote_history = []

    def predict_ensemble(self, image, categories):
        if image is None or self.model is None: return "", 0.0
        preds = []
        preds.append(self.model.predict(image.reshape(1,28,28,1), verbose=0)[0] * 1.0)
        preds.append(self.model.predict(np.fliplr(image).reshape(1,28,28,1), verbose=0)[0] * 0.3)
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            shifted = np.roll(np.roll(image, dx, axis=1), dy, axis=0)
            preds.append(self.model.predict(shifted.reshape(1,28,28,1), verbose=0)[0] * 0.15)
        avg = np.mean(preds, axis=0); avg = avg/avg.sum()
        best = np.argmax(avg)
        return categories[best] if best < len(categories) else "", float(avg[best])

    def predict_with_voting(self, image, categories):
        word, conf = self.predict_ensemble(image, categories)
        self.vote_history.append((word, conf))
        if len(self.vote_history) > 5: self.vote_history = self.vote_history[-5:]
        if len(self.vote_history) >= 3:
            votes = {}
            for w,c in self.vote_history: votes[w] = votes.get(w,0)+c
            bw = max(votes, key=votes.get)
            return bw, votes[bw]/len(self.vote_history)
        return word, conf

    def reset(self): self.vote_history = []

class AIManager:
    """100% OFFLINE AI Manager."""
    def __init__(self, base_model=None, categories=None):
        print("\n" + "="*50)
        print("KHOI TAO AI ASSISTANT (100% Offline)")
        print("="*50)
        self.categories = categories or []
        self.difficulty = SmartDifficulty()
        self.ensemble = EnsemblePredictor(base_model) if base_model else None
        print("  Chatbot + Goi y ve       OK")
        print("  Goi y thong minh         OK")
        print("  Ensemble nhan dien       " + ("OK" if self.ensemble else "NO MODEL"))
        print("  Cau vi du da dang        OK")
        print("="*50 + "\n")

    def get_explanation(self, word):
        return WORD_DATABASE.get(word, {}).get("explain", "Tu '{}' - Hay ve nhieu lan!".format(word))
    def get_drawing_hint(self, word):
        return WORD_DATABASE.get(word, {}).get("hint", "Hay ve '{}' don gian nhat!".format(word))
    def get_random_example(self, word):
        exs = WORD_DATABASE.get(word, {}).get("examples", [])
        return random.choice(exs) if exs else (None, None)
    def record_result(self, word, correct, time_taken, confidence):
        self.difficulty.record_result(word, correct, time_taken, confidence)
    def sort_words(self, word_list): return self.difficulty.sort_words(word_list)
    def should_hint(self, word, elapsed, has_content): return self.difficulty.should_hint(word, elapsed, has_content)
    def get_encouragement(self, correct, streak): return self.difficulty.get_encouragement(correct, streak)
    def get_difficulty_label(self): return self.difficulty.get_difficulty_label()
    def predict_enhanced(self, image):
        if self.ensemble and image is not None: return self.ensemble.predict_with_voting(image, self.categories)
        return "", 0.0
    def on_new_word(self):
        if self.ensemble: self.ensemble.reset()
