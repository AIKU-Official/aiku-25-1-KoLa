from transformers import AutoTokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class T5TextToLatex(nn.Module):
    def __init__(self, model_correct, model_tolatex, tokenizer):
        super().__init__()
        self.model_correct = model_correct
        self.model_tolatex = model_tolatex
        self.tokenizer = tokenizer
    
    def forward(self, noisy_inputs, return_outputs = True):
        # 텍스트 교정 (noisy_inputs -> corrected_text)
        correction_encodings = self.tokenizer(noisy_inputs, padding=True, truncation=True, max_length=128, return_tensors="pt")
        correction_input_ids = correction_encodings.input_ids.to(self.model_correct.device)
        correction_attention_mask = correction_encodings.attention_mask.to(self.model_correct.device)
        
        # 교정 모델을 통한 출력 생성 (학습/추론 모드 모두)
        with torch.no_grad():
            corrected_ids = self.model_correct.generate(
                input_ids=correction_input_ids,
                attention_mask=correction_attention_mask,
                repetition_penalty = 1.5,
                max_length=128
            )
            corrected_texts = self.tokenizer.batch_decode(corrected_ids, skip_special_tokens=True)
        
        # LaTeX 변환 모델 (corrected_text -> latex)
        latex_encodings = self.tokenizer(corrected_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        latex_input_ids = latex_encodings.input_ids.to(self.model_tolatex.device)
        latex_attention_mask = latex_encodings.attention_mask.to(self.model_tolatex.device)
        

        with torch.no_grad():
            latex_ids = self.model_tolatex.generate(
                input_ids=latex_input_ids,
                attention_mask=latex_attention_mask,
                repetition_penalty = 1.5,
                max_length=128
            )
            latex_outputs = self.tokenizer.batch_decode(latex_ids, skip_special_tokens=True)

        return corrected_texts, latex_outputs
    
def transcribe_one(file_path: str,model_name: str = "openai/whisper-large-v3-turbo",device: torch.device = None) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()

    # 리샘플링
    speech_array, sampling_rate = torchaudio.load(file_path)                  
    resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
    speech = resampler(speech_array).squeeze()                            
    speech = speech.cpu().numpy()                                           

    # 전처리 → 입력 피처 생성
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt",return_attention_mask=True,task="transcribe")        # CPU tensor
    input_features = inputs.input_features.to(device) 
    attention_mask = inputs.attention_mask.to(device)
    
    forced_ids = processor.get_decoder_prompt_ids(
        task="transcribe"
    )

    # 4) generate → decode
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features, 
            attention_mask=attention_mask,
            forced_decoder_ids=forced_ids,)
        
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    return transcription

def main():
    audio_path = "latex_data/3_audio/tts_mathbridge_00/29.mp3"
    error_texts = transcribe_one(audio_path)
    # error_texts = ["n 프라임","n 푸라임"] # list str 형태.
    model_path = "ke-t5-base"
    
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_correct = T5ForConditionalGeneration.from_pretrained(model_path)
    model_tolatex = T5ForConditionalGeneration.from_pretrained(model_path)
    
    corrector_weight = "corrector_best_model.pt"
    translator_weight = "translator_best_model.pt"
    
    corrector_checkpoint = torch.load(corrector_weight, map_location=device,weights_only=False)
    translator_checkpoint = torch.load(translator_weight, map_location=device,weights_only=False)
    
    model_correct.load_state_dict(corrector_checkpoint['model_state_dict'])
    model_tolatex.load_state_dict(translator_checkpoint['model_state_dict'])
    
    model_correct.eval()
    model_tolatex.eval()
    
    model = T5TextToLatex(model_correct, model_tolatex, tokenizer)
    model.to(device)
    model.eval()

    corrected_texts, pred_latexes = model(error_texts, return_outputs=True)
    
    print("error text : ", error_texts[0])
    print("corrected_texts : ",corrected_texts[0])
    print("pred_latexes : ",pred_latexes[0])
    
    
if __name__ == "__main__":
    main()
