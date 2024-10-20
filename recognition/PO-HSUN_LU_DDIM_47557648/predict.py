from train import *

def main():
    # hyperparameter
    LR = 7e-5
    BATCH_SIZE = 8
    IMG_SIZE = 128
    FILTER_SIZE = 64
    TOTAL_ITERATION = 100000
    SAVE_N_ITERATION = 10000
    CKPT_DIR = './model_weight/'

    dataloader = None
    # Training & plot image
    # model will load the weight if load_path is not None
    load_path = "Put your model weight path here !!!"
    model = U_net(FILTER_SIZE)
    trainer = Trainer(model, dataloader, DDPM, CKPT_DIR, load_path=load_path, 
                  total_step=TOTAL_ITERATION, save_n_step=SAVE_N_ITERATION, lr=LR)
    
    # This function will plot and save the rsult as jpg image
    # by setting cond = 'NC' or "AD" can swich to different condition
    trainer.ddim_sample(cond = 'NC')

if __name__ == '__main__':
    main()