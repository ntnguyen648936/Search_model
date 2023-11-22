from nltk.tokenize import sent_tokenize
from collections import Counter
def stack_text_sections(text, max_tokens_per_section):
    """
    Chồng chất các phần văn bản có độ dài tối đa là max_tokens_per_section.
    """
    sections = []
    tokens = text.split()

    current_section = []
    current_tokens_count = 0

    for token in tokens:
        current_tokens_count += 1
        current_section.append(token)

        if current_tokens_count >= max_tokens_per_section:
            sections.append(" ".join(current_section))
            current_section = []
            current_tokens_count = 0

    if current_section:
        sections.append(" ".join(current_section))

    return sections

def extract_related_sentences(content, user_input, num_sentences=3):
    sentences = sent_tokenize(content)
    related_sentences = []

    for sentence in sentences:
        if user_input in sentence:
            related_sentences.append(sentence)
        if len(related_sentences) >= num_sentences:
            break

    return related_sentences

def majority_vote(categories):

    # Chuyển đổi danh sách categories thành tuple để làm cho nó hashable
    categories_tuple = tuple(map(tuple, categories))
    
    # Đếm số lần xuất hiện của mỗi nhãn category
    count = Counter(categories_tuple)
    
    # Chọn nhãn có số lần xuất hiện nhiều nhất
    category = count.most_common(1)[0][0]
    
    return category



