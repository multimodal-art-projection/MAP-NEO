import fasttext

model_path = 'model_offcial.ftz'
model = fasttext.load_model(model_path)  

def preprocess_text(text):
    return text.replace('\n', '\\n').replace('__label__', '')

# Modified filter_text function
def compute_confidence(text):
    if text == None:
        return 0.0
    data_text = preprocess_text(text)
    label, confidence = model.predict(data_text)
    if label[0] == '__label__1':
        return float(confidence[0])
    else:
        return 0.0