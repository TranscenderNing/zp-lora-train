# training 
python run_clm_sft_with_peft_1.py

# merge lora adaptor to model
pyton merge.py

# get model infer results
python alpaca_eval.py
