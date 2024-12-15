import sys
import subprocess
import argparse
""" 
Strucutre of arguments: 
Architecutre
    ASK:
        Embedding

        Number of layers
        Kernel size
        etc.
"""

scriptdir = './scripts/MBE_tunning_TOMAS'
# procdir = './out/modelling/processed'

arch = sys.argv[1]
args_size = len(sys.argv)


# parser = argparse.ArgumentParser(description='Run different scripts with arguments.')
# subparsers = parser.add_subparsers(dest='script', help='Choose which script to run')
# parser_script = subparsers.add_parser('script1', help='Run script1')
# parser_script.add_argument('script1_args', nargs=argparse.REMAINDER, help='Arguments for script1')
    
# args = parser.parse_args()

if arch == 'FF':
    # NEEDED: embedding, number of layers,
    """
    format: embedding, number of layers, weight decay,hidden layer size variation format
    """
    params = ['ESM', 2, 1e-5, 'None']

    if args_size < 4:
        print("ERROR: not enough parameters")
        sys.exit()
    for i, arg in enumerate(sys.argv):
        if i != 0 and i != 1:
            params[i-2] = arg
    
    params = [str(p) for p in params]
    subprocess.run(["python3", f"{scriptdir}/mbe-feedforward.py"] + params)
    
    

elif arch == 'LSTM':
    h_size = 3
    n_layers = 1
    wd = 1e-5
    if args_size < 4:
        print("ERROR: not enough parameters")
        sys.exit()
    elif args_size == 4:
        h_size = sys.argv[2]
        n_layers = sys.argv[3]
    else:
        h_size = sys.argv[2]
        n_layers = sys.argv[3]
        wd = sys.argv[4]
    
    args = [h_size, n_layers, wd]
    args = [str(a) for a in args]

    subprocess.run(["python3", f"{scriptdir}/mbe-ESM2-LSTM.py"] + args)
    
elif arch == 'CNN':
    
    """
    N_LAYERS = int(sys.argv[1])
    OUT_CHANNELS = ast.literal_eval(sys.argv[2])
    KERNEL_SIZE = int(sys.argv[3])
    WEIGHT_DECAY = float(sys.argv[4])
    STRIDE = int(sys.arv[5])
    PADDING = int(sys.argv[6])
    DIALATION = int(sys.argv[7])
    """
    args = [2, (128, 64), 3, 1e-5, 1, 0, 0]
    for i in range(len(sys.argv) - 2):
        args[i] = sys.argv[i + 2]
    
    args = [str(a) for a in args]
    subprocess.run(["python3", f"{scriptdir}/mbe-CNN.py"] + args)


print("made it to the end!")