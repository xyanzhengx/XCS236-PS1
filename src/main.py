import sys
import datasets.encoder as enc
from model import GPT2, load_weight
from datasets.encoder import get_codec
from datasets.neurips_dataset import NIPS2015Dataset
from utils import *
import requests
import torch
import pickle as pkl
from tqdm import trange
import numpy as np
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import importlib.util

# directory to save data
save_dir = "submission"

# Check if submission module is present.  If it is not, then main() will not be executed.
use_submission = importlib.util.find_spec('submission') is not None
if use_submission:
  from submission import classification, log_likelihood, sample

def downloadGPT2Checkpoint():
    if not os.path.exists('./gpt2-pytorch_model.bin'):
        print("Downloading GPT-2 checkpoint...")
        url = 'https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin'
        r = requests.get(url, allow_redirects=True)
        open('gpt2-pytorch_model.bin', 'wb').write(r.content)
    assert os.path.exists("./gpt2-pytorch_model.bin")

def setup(device):
    config = parse_config()
    downloadGPT2Checkpoint()

    np.random.seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.manual_seed(config.seed)

    paper_dataset = NIPS2015Dataset(data_folder='datasets')

    codec = get_codec()
    model = GPT2(config)
    if not os.path.exists('gpt2-pytorch_model.bin'):
        print("Downloading GPT-2 checkpoint...")
        url = 'https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin'
        r = requests.get(url, allow_redirects=True)
        open('gpt2-pytorch_model.bin', 'wb').write(r.content)

    model = load_weight(model, torch.load('gpt2-pytorch_model.bin', map_location=device))
    model = model.to(device)
    model.eval()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return codec, model, paper_dataset, config


def plot_log_p(filename, codec, model, device):
    with open(os.path.join('datasets', filename + '.pkl'), 'rb') as f:
        lls = []
        data = pkl.load(f)
        for i in trange(len(data)):
            text = data[i]
            text = codec.encode(text).to(device)
            ## TODO: complete the code in the function `log_likelihood`
            lls.append(log_likelihood(model, text))
        lls = np.asarray(lls)

    with open(os.path.join(save_dir, filename + '_raw.pkl'), 'wb') as f:
        pkl.dump(lls, f, protocol=pkl.HIGHEST_PROTOCOL)

    plt.figure()
    plt.hist(lls)
    plt.xlabel('Log-likelihood')
    plt.xlim([-600, 0])
    plt.ylabel('Counts')
    plt.title(filename)
    plt.savefig(os.path.join(save_dir, filename + '.png'), bbox_inches='tight')
    plt.show()
    plt.close()
    print("# Figure written to %s.png." % filename)


def main():
    args = parse_args()
    if args.cache:
        downloadGPT2Checkpoint()
        return
    
    # allowing for GPU and CPU support
    # NOTE: we are not supporting `torch.device("mps")` as it trains slower than cpu on Apple Silicon devices 
    if args.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Torch Device: {device}")

    codec, model, paper_dataset, config = setup(device)

    print("========= Question 6c =========")
    paper_iter = iter(paper_dataset)
    with open(os.path.join(save_dir, 'samples.txt'), 'w', encoding='utf-8') as f:
        for i in range(5):
            ## Use paper abstracts as the starting text
            start_text = next(paper_iter)['abstract'][:100]
            start_text = codec.encode(start_text).to(device)
            ## TODO: Complete the code for the sample function within submission/sample.py
            text = sample(model, start_text, config, length=config.n_ctx // 2)
            ## Decode samples
            text = codec.decode(text.tolist()[0])
            f.write('=' * 50 + " SAMPLE_{} ".format(i) + '=' * 50 + '\n')
            f.write(text + '\n')
    print("# Samples written to samples.txt.")

    print("========= Question 6d =========")
    ## TODO: Complete the code for the log_likelihood function within submission/likelihood.py
    plot_log_p('random', codec, model, device)
    plot_log_p('shakespeare', codec, model, device)
    plot_log_p('neurips', codec, model, device)

    print("========= Question 6e =========")
    with open(os.path.join('datasets', 'snippets.pkl'), 'rb') as f:
        snippets = pkl.load(f)
    lbls = []
    for snippet in snippets:
        ## TODO: Complete the code for the classification function within submission/classifier.py
        lbls.append(classification(model, codec.encode(snippet).to(device)))

    with open(os.path.join(save_dir, "classification.pkl"), 'wb') as f:
        pkl.dump(lbls, f, protocol=pkl.HIGHEST_PROTOCOL)
    print("# Classification completed.")

    print("========= Question 6f =========")
    paper_iter = iter(paper_dataset)
    with open(os.path.join(save_dir, 'samples_temperature0-95.txt'), 'w', encoding='utf-8') as f:
        for i in range(5):
            ## Use paper abstracts as the starting text
            start_text = next(paper_iter)['abstract'][:100]
            start_text = codec.encode(start_text).to(device)
            ## TODO: Complete the code for the temperature_scale function within submission/sample.py when temperature_horizon=1
            text = sample(model, start_text, config, length=config.n_ctx // 2, temperature=0.95, temperature_horizon=1)
            ## Decode samples
            text = codec.decode(text.tolist()[0])
            f.write('=' * 50 + " SAMPLE_{} x".format(i) + '=' * 50 + '\n')
            f.write(text + '\n')
    print("# Samples written to samples_temperature0-95.txt.")

    print("========= Question 6h =========")
    paper_iter = iter(paper_dataset)
    with open(os.path.join(save_dir, 'samples_temperature0-95_horizon2.txt'), 'w', encoding='utf-8') as f:
        for i in range(1):
            ## Use paper abstracts as the starting text
            start_text = next(paper_iter)['abstract'][:100]
            start_text = codec.encode(start_text).to(device)
            ## TODO: Complete the code for the temperature_scale function within submission/sample.py when temperature_horizon=2
            text = sample(model, start_text, config, length=100, temperature=0.95, temperature_horizon=2)
            ## Decode samples
            text = codec.decode(text.tolist()[0])
            f.write('=' * 50 + " SAMPLE_{} ".format(i) + '=' * 50 + '\n')
            f.write(text + '\n')
    print("# Samples written to samples_temperature0-95_horizon2.txt.")
    return 0

if __name__ == '__main__':
    sys.exit(main())
