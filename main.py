import itertools

from nltk import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

question_tokenizer = AutoTokenizer.from_pretrained('t5-small-qg-hl')
question_model = AutoModelForSeq2SeqLM.from_pretrained('t5-small-qg-hl')

answer_tokenizer = AutoTokenizer.from_pretrained('t5-small-qa-qg-hl')
answer_model = AutoModelForSeq2SeqLM.from_pretrained('t5-small-qa-qg-hl')

question_model.to('cuda')
answer_model.to('cuda')

text = '''
Albert Einstein, (born March 14, 1879, Ulm, Württemberg, Germany—died April 18, 1955, Princeton, New Jersey, U.S.), German-born physicist who developed the special and general theories of relativity and won the Nobel Prize for Physics in 1921 for his explanation of the photoelectric effect. Einstein is generally considered the most influential physicist of the 20th century. Einstein’s parents were secular, middle-class Jews. His father, Hermann Einstein, was originally a featherbed salesman and later ran an electrochemical factory with moderate success. His mother, the former Pauline Koch, ran the family household. He had one sister, Maria (who went by the name Maja), born two years after Albert. Einstein would write that two “wonders” deeply affected his early years. The first was his encounter with a compass at age five. He was mystified that invisible forces could deflect the needle. This would lead to a lifelong fascination with invisible forces. The second wonder came at age 12 when he discovered a book of geometry, which he devoured, calling it his “sacred little geometry book.”
'''

sentences = sent_tokenize(text)

texts = ['extract answers: ' + text for _ in range(len(sentences))]

for i, sentence in enumerate(sentences):
    texts[i] = texts[i].replace(sentence, ' <hl> ' + sentence.strip() + ' <hl> ')

inputs = answer_tokenizer.batch_encode_plus(texts,
                                            max_length=512,
                                            add_special_tokens=True,
                                            truncation=True,
                                            padding='max_length',
                                            pad_to_max_length=True,
                                            return_tensors='pt')
outputs = answer_model.generate(input_ids=inputs['input_ids'].to('cuda'),
                                attention_mask=inputs['attention_mask'].to('cuda'),
                                max_length=32)

answers = [answer_tokenizer.decode(output, skip_special_tokens=False)[:-6].strip().split(' <sep> ') for output in outputs]

inputs = []
for i in range(len(texts)):
    texts[i] = texts[i].replace('<hl>', '')
    for answer in answers[i]:
        inputs.append(texts[i].replace(answer, '<hl>' + answer + '<hl>'))

inputs = question_tokenizer.batch_encode_plus(inputs,
                                              max_length=512,
                                              add_special_tokens=True,
                                              truncation=True,
                                              padding='max_length',
                                              pad_to_max_length=True,
                                              return_tensors='pt')
outputs = question_model.generate(input_ids=inputs['input_ids'].to('cuda'),
                                  attention_mask=inputs['attention_mask'].to('cuda'),
                                  max_length=32,
                                  num_beams=4)

questions = [question_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

for question, answer in zip(questions, list(itertools.chain.from_iterable(answers))):
    print((question, answer))
