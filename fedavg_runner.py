# python3 fedavg_runner.py --dataset=Movies_and_TV --batch_size=128 --lr=0.001 --maxlen=50 --hidden_units=50 --num_blocks=2 --num_epochs=200 --num_heads=1 --dropout_rate=0.5 --l2_emb=0.0 --device=cuda --inference_only=False --state_dict_path=None

from src.model import FARF
import argparse
import json

from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.script_runner import ScriptRunner
from src.utils import data_partition


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Movies_and_TV')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, action='store_true')
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--num_clients', default=3, type=int)
parser.add_argument('--num_rounds', default=2, type=int)


arg = parser.parse_args()

if __name__ == "__main__":
    n_clients = arg.num_clients
    num_rounds = arg.num_rounds
    train_script = "src/client_script.py"
    
    if(arg.dataset=='Movies_and_TV'):
        text_file_path=f'./data/amazon/Movies_and_TV.txt'
    elif(arg.dataset=='CiteULike'):
        text_file_path=f'./data/CiteULike/CiteULikeoutput.txt'
    elif(arg.dataset=='Movie_Lens'):
        text_file_path=f'./data/Movie_Lens/Movie_Lens_output2.txt'
    
    [_, _, _, usernum, itemnum] = data_partition(f'{arg.dataset}',text_file_path)
    # usernum = 311143
    # usernum = 6038
    # itemnum = 86678
    # itemnum=3627

    arg_handler = json.dumps(arg.__dict__)

    l = arg_handler.replace('"', '&')[1:-1]
    model = FARF(usernum, itemnum, l)

    job = FedAvgJob(
        name="sasrec_fedavg", n_clients=n_clients, num_rounds=num_rounds, initial_model=model
    )
    print("Job created")
    # Add clients
    for i in range(n_clients):
        print("Adding client{}".format(i+1))
        executor = ScriptRunner(
            script=train_script,
            script_args=f"--dataset={arg.dataset} --batch_size={arg.batch_size} --num_epochs={arg.num_epochs}", # f"--dataset=Movies_and_TV --batch_size=128 --lr=0.001 --maxlen=50 --hidden_units=50 --num_blocks=2 --num_epochs=200 --num_heads=1 --dropout_rate=0.5 --l2_emb=0.0 --device=cuda --inference_only=False --state_dict_path=None"
        )
        print("Executor created")
        job.to(executor, f"site-{i+1}")
        print("Added client{}".format(i+1))
    print("All clients added")
    job.simulator_run("tmp/nvflare/jobs/workdir", gpu="0")
    print("Job run successfully")
