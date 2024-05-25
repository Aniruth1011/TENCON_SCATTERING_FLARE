import argparse

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='De Flarer Training')
        self.initialize()

    def initialize(self):
        # Data parameters
        self.parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
        self.parser.add_argument('--no_of_workers', type=int, default=8, help='Number of workers for DataLoader')

        # Training parameters
        self.parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
        self.parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
        self.parser.add_argument('--accumulation_steps', type=int, default=8, help='Number of steps to accumulate gradients')
        self.parser.add_argument('--threshold_learning_rate' , type = float , default=0.1 , help = 'Learning rate of Masking Threshold')
        self.parser.add_argument('--inpaint_learning_rate', type = float  , default = 0.0003 , help = 'Learning Rate for the Gan Model')

        # Device
        self.parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')

        # Model parameters
        self.parser.add_argument("--model", type=str, default="aotgan", help="model name")

        self.parser.add_argument('--common_model_path', type=str, default='ckpt/together', help='Path to save trained model')
        self.parser.add_argument('--lensformer_model_path', type=str, default='ckpt/lensformer', help='Path to save trained model')
        self.parser.add_argument('--gan_model_path', type=str, default='ckpt/gan', help='Path to save trained model')
        self.parser.add_argument('--promptgan_model_path', type=str, default='ckpt/promptgan', help='Path to save trained model')

        self.parser.add_argument('--save_every' , type = int , default = 10 , help = 'Model Checkpoints saving')
        self.parser.add_argument('--with_prompts' , type = bool , default = True , help = 'Use Learnable Prompting')
        self.parser.add_argument("--block_num", type=int, default=8, help="number of AOT blocks")
        self.parser.add_argument("--rates", type=str, default="1+2+4+8", help="dilation rates used in AOT block")
        self.parser.add_argument("--beta1", type=float, default=0.5, help="beta1 in optimizer")
        self.parser.add_argument("--beta2", type=float, default=0.999, help="beta2 in optimizer")
        self.parser.add_argument("--lrg", type=float, default=1e-4, help="learning rate for generator")
        self.parser.add_argument("--rec_loss", type=str, default="1*L1+250*Style+0.1*Perceptual", help="losses for reconstruction")

        # DataLoader parameters
        self.parser.add_argument('--train_dataset_path', type=str, default='data', help='Path to training dataset')
        self.parser.add_argument('--image_size' , type = int , default = 256 , help = 'Size to which image gets resized')

        #Pre Trained Model 
        self.parser.add_argument('--pretrained' , type = str , default=r'places_pretrained' , help = 'Path for the pre trained model')

    def parse(self):
        return self.parser.parse_args()

options = Options().parse()
