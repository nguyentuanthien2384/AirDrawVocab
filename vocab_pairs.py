"""
vocab_pairs.py — Nguồn từ vựng TRUNG TÂM cho AirDrawVocab (mở rộng 40 lớp).

Mỗi lớp QuickDraw kèm: nghĩa tiếng Việt, phiên âm IPA, câu ví dụ (EN + VI),
và gợi ý cách vẽ. Dùng chung cho train (config.CATEGORIES), backend web,
notebook và phần demo — sửa 1 nơi, cả dự án cập nhật theo.

Thứ tự khóa = thứ tự nhãn của model. KHÔNG đổi thứ tự sau khi đã train,
nếu không sẽ lệch nhãn (đây từng là nguyên nhân lỗi "đoán toàn apple").
"""

VOCAB = {
    # ---------------- 19 lớp gốc ----------------
    "apple":     {"vi": "quả táo",    "ipa": "/ˈæp.əl/",      "ex": "I eat an apple every morning.",        "ex_vi": "Tôi ăn một quả táo mỗi sáng.",          "hint": "Vẽ vòng tròn, thêm cuống và lá nhỏ phía trên."},
    "baseball":  {"vi": "bóng chày",  "ipa": "/ˈbeɪs.bɔːl/",  "ex": "He plays baseball after school.",      "ex_vi": "Anh ấy chơi bóng chày sau giờ học.",     "hint": "Vẽ vòng tròn, thêm 2 đường cong khâu bóng."},
    "book":      {"vi": "quyển sách", "ipa": "/bʊk/",         "ex": "This book is very interesting.",       "ex_vi": "Quyển sách này rất thú vị.",             "hint": "Vẽ hình chữ nhật, kẻ gáy sách ở giữa."},
    "bowtie":    {"vi": "nơ bướm",    "ipa": "/ˈboʊ.taɪ/",    "ex": "He wears a bowtie at the party.",      "ex_vi": "Anh ấy đeo nơ bướm ở bữa tiệc.",         "hint": "Vẽ 2 tam giác chạm nhau, thêm nút ở giữa."},
    "diamond":   {"vi": "kim cương",  "ipa": "/ˈdaɪ.mənd/",   "ex": "The diamond is very shiny.",           "ex_vi": "Viên kim cương rất lấp lánh.",           "hint": "Vẽ hình thoi: đỉnh trên, dưới và hai góc ngang."},
    "dog":       {"vi": "con chó",    "ipa": "/dɒɡ/",         "ex": "The dog is friendly.",                 "ex_vi": "Con chó rất thân thiện.",                "hint": "Vẽ đầu tròn, 2 tai, mắt và mũi."},
    "door":      {"vi": "cánh cửa",   "ipa": "/dɔːr/",        "ex": "Please close the door.",               "ex_vi": "Làm ơn đóng cửa lại.",                   "hint": "Vẽ hình chữ nhật đứng, thêm tay nắm tròn."},
    "envelope":  {"vi": "phong bì",   "ipa": "/ˈen.və.loʊp/", "ex": "She puts the letter in an envelope.",  "ex_vi": "Cô ấy bỏ lá thư vào phong bì.",          "hint": "Vẽ hình chữ nhật ngang, thêm nét chữ V."},
    "eye":       {"vi": "con mắt",    "ipa": "/aɪ/",          "ex": "My eye is blue.",                      "ex_vi": "Mắt tôi màu xanh.",                      "hint": "Vẽ oval nằm ngang, thêm tròng và con ngươi."},
    "fish":      {"vi": "con cá",     "ipa": "/fɪʃ/",         "ex": "The fish swims in the water.",         "ex_vi": "Con cá bơi trong nước.",                 "hint": "Vẽ thân oval, đuôi tam giác và mắt."},
    "hat":       {"vi": "cái mũ",     "ipa": "/hæt/",         "ex": "I wear a hat on sunny days.",          "ex_vi": "Tôi đội mũ vào ngày nắng.",              "hint": "Vẽ nửa vòng tròn, thêm vành ngang."},
    "leaf":      {"vi": "chiếc lá",   "ipa": "/liːf/",        "ex": "A leaf falls from the tree.",          "ex_vi": "Một chiếc lá rơi từ trên cây.",          "hint": "Vẽ oval nhọn, thêm gân lá ở giữa."},
    "lightning": {"vi": "tia sét",    "ipa": "/ˈlaɪt.nɪŋ/",   "ex": "Lightning appears during the storm.",  "ex_vi": "Tia sét xuất hiện trong cơn bão.",       "hint": "Vẽ đường zigzag nhọn từ trên xuống."},
    "moon":      {"vi": "mặt trăng",  "ipa": "/muːn/",        "ex": "The moon is bright tonight.",          "ex_vi": "Mặt trăng tối nay rất sáng.",            "hint": "Vẽ trăng lưỡi liềm bằng 2 đường cong."},
    "pants":     {"vi": "quần dài",   "ipa": "/pænts/",       "ex": "These pants are black.",               "ex_vi": "Cái quần này màu đen.",                  "hint": "Vẽ cạp quần và 2 ống quần."},
    "scissors":  {"vi": "cái kéo",    "ipa": "/ˈsɪz.ɚz/",     "ex": "I cut paper with scissors.",           "ex_vi": "Tôi cắt giấy bằng kéo.",                 "hint": "Vẽ chữ X, thêm 2 vòng tròn tay cầm."},
    "square":    {"vi": "hình vuông", "ipa": "/skwer/",       "ex": "This is a red square.",                "ex_vi": "Đây là một hình vuông màu đỏ.",          "hint": "Vẽ 4 cạnh đều, khép kín."},
    "star":      {"vi": "ngôi sao",   "ipa": "/stɑːr/",       "ex": "A star shines in the sky.",            "ex_vi": "Một ngôi sao tỏa sáng trên trời.",       "hint": "Vẽ 5 đỉnh nhọn nối liền."},
    "t-shirt":   {"vi": "áo thun",    "ipa": "/ˈtiː.ʃɜːrt/",  "ex": "I like this t-shirt.",                 "ex_vi": "Tôi thích cái áo thun này.",             "hint": "Vẽ thân áo, 2 tay áo và cổ áo."},

    # ---------------- 21 lớp mới (mở rộng) ----------------
    "cat":       {"vi": "con mèo",    "ipa": "/kæt/",         "ex": "The cat sleeps on the sofa.",          "ex_vi": "Con mèo ngủ trên ghế sofa.",             "hint": "Vẽ đầu tròn, 2 tai nhọn, ria mép."},
    "sun":       {"vi": "mặt trời",   "ipa": "/sʌn/",         "ex": "The sun rises in the east.",           "ex_vi": "Mặt trời mọc ở hướng đông.",             "hint": "Vẽ vòng tròn, thêm các tia nắng quanh."},
    "tree":      {"vi": "cái cây",    "ipa": "/triː/",        "ex": "A bird sits in the tree.",             "ex_vi": "Một con chim đậu trên cây.",             "hint": "Vẽ thân cây, thêm tán lá tròn phía trên."},
    "flower":    {"vi": "bông hoa",   "ipa": "/ˈflaʊ.ɚ/",     "ex": "She picked a red flower.",             "ex_vi": "Cô ấy hái một bông hoa đỏ.",             "hint": "Vẽ tâm tròn, thêm các cánh hoa quanh."},
    "cloud":     {"vi": "đám mây",    "ipa": "/klaʊd/",       "ex": "A white cloud floats in the sky.",     "ex_vi": "Một đám mây trắng trôi trên trời.",      "hint": "Vẽ vài đường cong tròn nối thành cụm."},
    "umbrella":  {"vi": "cái ô",      "ipa": "/ʌmˈbrel.ə/",   "ex": "I use an umbrella when it rains.",     "ex_vi": "Tôi dùng ô khi trời mưa.",               "hint": "Vẽ vòm bán nguyệt, thêm cán cong bên dưới."},
    "key":       {"vi": "chìa khóa",  "ipa": "/kiː/",         "ex": "I lost my house key.",                 "ex_vi": "Tôi làm mất chìa khóa nhà.",             "hint": "Vẽ vòng tròn đầu, thân dài và răng khóa."},
    "cup":       {"vi": "cái cốc",    "ipa": "/kʌp/",         "ex": "Pour the tea into the cup.",           "ex_vi": "Rót trà vào cốc.",                       "hint": "Vẽ thân cốc hình thang, thêm quai bên."},
    "clock":     {"vi": "đồng hồ",    "ipa": "/klɒk/",        "ex": "The clock shows three o'clock.",       "ex_vi": "Đồng hồ chỉ ba giờ.",                    "hint": "Vẽ vòng tròn, thêm 2 kim đồng hồ."},
    "car":       {"vi": "xe hơi",     "ipa": "/kɑːr/",        "ex": "My father drives a car.",              "ex_vi": "Bố tôi lái xe hơi.",                     "hint": "Vẽ thân xe, 2 bánh tròn và cửa kính."},
    "bicycle":   {"vi": "xe đạp",     "ipa": "/ˈbaɪ.sɪ.kəl/", "ex": "She rides her bicycle to school.",     "ex_vi": "Cô ấy đạp xe đến trường.",               "hint": "Vẽ 2 bánh tròn, khung và ghi đông."},
    "airplane":  {"vi": "máy bay",    "ipa": "/ˈer.pleɪn/",   "ex": "The airplane flies above the clouds.", "ex_vi": "Máy bay bay trên những đám mây.",        "hint": "Vẽ thân dài, 2 cánh và đuôi."},
    "house":     {"vi": "ngôi nhà",   "ipa": "/haʊs/",        "ex": "They live in a big house.",            "ex_vi": "Họ sống trong một ngôi nhà lớn.",        "hint": "Vẽ hình vuông thân nhà, thêm mái tam giác."},
    "banana":    {"vi": "quả chuối",  "ipa": "/bəˈnæn.ə/",    "ex": "A monkey eats a banana.",              "ex_vi": "Một con khỉ ăn quả chuối.",              "hint": "Vẽ một đường cong dài hình lưỡi liềm."},
    "ice cream": {"vi": "kem",        "ipa": "/ˈaɪs ˌkriːm/", "ex": "I love chocolate ice cream.",          "ex_vi": "Tôi thích kem sô-cô-la.",                "hint": "Vẽ ốc quế tam giác, thêm viên kem tròn trên."},
    "cake":      {"vi": "bánh ngọt",  "ipa": "/keɪk/",        "ex": "We eat cake on birthdays.",            "ex_vi": "Chúng tôi ăn bánh vào sinh nhật.",       "hint": "Vẽ thân bánh, thêm nến và kem trên."},
    "candle":    {"vi": "cây nến",    "ipa": "/ˈkæn.dəl/",    "ex": "She blew out the candle.",             "ex_vi": "Cô ấy thổi tắt cây nến.",                "hint": "Vẽ thân nến đứng, thêm ngọn lửa nhỏ."},
    "guitar":    {"vi": "đàn ghi-ta", "ipa": "/ɡɪˈtɑːr/",     "ex": "He plays the guitar very well.",       "ex_vi": "Anh ấy chơi ghi-ta rất hay.",            "hint": "Vẽ thân đàn hình số 8, thêm cần đàn dài."},
    "hammer":    {"vi": "cái búa",    "ipa": "/ˈhæm.ɚ/",      "ex": "He hits the nail with a hammer.",      "ex_vi": "Anh ấy đóng đinh bằng búa.",             "hint": "Vẽ cán dài, thêm đầu búa chữ nhật trên."},
    "bed":       {"vi": "cái giường", "ipa": "/bed/",         "ex": "I sleep in my bed at night.",          "ex_vi": "Tôi ngủ trên giường vào ban đêm.",       "hint": "Vẽ khung giường, thêm gối và đầu giường."},
    "chair":     {"vi": "cái ghế",    "ipa": "/tʃer/",        "ex": "Please sit on the chair.",             "ex_vi": "Mời ngồi lên ghế.",                      "hint": "Vẽ lưng ghế, mặt ngồi và 4 chân."},
}

CATEGORIES = list(VOCAB.keys())
NUM_CLASSES = len(CATEGORIES)

# Các dict tiện dùng (tương thích ngược với code cũ)
VI_MEANINGS = {k: v["vi"] for k, v in VOCAB.items()}
EXAMPLE_SENTENCES = {k: v["ex"] for k, v in VOCAB.items()}
EXAMPLE_SENTENCES_VI = {k: v["ex_vi"] for k, v in VOCAB.items()}
IPA = {k: v["ipa"] for k, v in VOCAB.items()}
DRAWING_HINTS = {k: v["hint"] for k, v in VOCAB.items()}


def translate(label: str) -> dict:
    """Trả về toàn bộ thông tin dịch nghĩa của 1 nhãn (dùng cho chatbot/UI)."""
    v = VOCAB.get(label)
    if not v:
        return {"label": label, "vi": label, "ipa": "", "ex": f"This is a {label}.",
                "ex_vi": "", "hint": ""}
    return {"label": label, **v}


if __name__ == "__main__":
    print(f"Tổng số lớp từ vựng: {NUM_CLASSES}")
    for k in CATEGORIES:
        print(f"  {k:11s} -> {VOCAB[k]['vi']:12s} {VOCAB[k]['ipa']}")
