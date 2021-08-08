import torch
from Node import Node, Global_Node
from Args import args_parser
from Data import Data
from utils import LR_scheduler, Recorder, Catfish, Summary
from Trainer import Trainer

# init args
args = args_parser()
args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
print('Running on', args.device)
Data = Data(args)
Train = Trainer(args)

# init nodes
Global_node = Global_Node(Data.test_all, args)
#Node_List = [Node(k, Data.train_loader[k], Data.test_loader, args) for k in range(args.node_num)]
#Catfish(Node_List, args)
Node_List=[]
recorder=[]
Model_List=['LeNet5', 'MLP', 'CNN', 'ResNet18','CNN','LeNet5', 'MLP', 'CNN', 'ResNet18','CNN']# local 模型不一样
for k in range(args.node_num):
    argss=args_parser()
    argss.device = torch.device(argss.device if torch.cuda.is_available() else 'cpu')
    argss.local_model=Model_List[k]
    Node_List.append(Node(k,Data.train_loader[k], Data.test_loader, argss))
    recorder.append(Recorder(argss))
    
# init variables
#recorder = Recorder(args)
#Summary(args)
# start
for rounds in range(args.R):
    print('===============The {:d}-th round==============='.format(rounds + 1))
    LR_scheduler(rounds, Node_List, args)
    for k in range(len(Node_List)):
       
        for epoch in range(args.E):
            Train(Node_List[k],Node_List)
            recorder[k].validate(Node_List[k])
        recorder[k].printer(Node_List[k])

#    recorder.validate(Global_node)
#    recorder.printer(Global_node)
#recorder.finish()
#Summary(args)
for k in range(len(Node_List)):
    recorder[k].finish(Node_List[k])
