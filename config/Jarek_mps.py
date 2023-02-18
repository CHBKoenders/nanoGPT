import torch._dynamo as dynamo
dynamo.config.suppress_errors = True

out_dir = '../Out/Jarek/'
eval_interval = 5
eval_iters = 40
wandb_log = True # feel free to turn on
wandb_project = 'JarekGPT'

device = "mps"

dataset = 'Jarek'
init_from = 'gpt2'
# init_from = 'resume'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 210

# finetune at constant LR
learning_rate = 1.2e-5
decay_lr = False
