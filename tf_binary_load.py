import argparse
import os
import time
#from experiments import *


import tensorflow as tf
import json
import random    
import resource

import torchvision.transforms as transforms
import torch
from torchvision import datasets as torchdatasets
import numpy as np

from tensorflow.python.tools import freeze_graph
BEST_MODEL_DEFAULT_FILE_NAME = 'model_best.pth.tar'
CONFIG_DEFAULT_FILE_NAME = 'config.json'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    #print(output, target)

    _, pred = output.float().topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    #print(output)
    #print(pred, target)
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0]

        

def load_data(config):
    if config["name"] in {"mnist"}:
        train_loader = torch.utils.data.DataLoader(
            torchdatasets.MNIST(config["data"]["data_dir"], train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((config["data"]["mean_norm"],), (config["data"]["var_norm"],))
                           ])),
            batch_size=config["train"]["batch_size"], shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            torchdatasets.MNIST(config["data"]["data_dir"], train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((config["data"]["mean_norm"],), (config["data"]["var_norm"],))
                           ])),
            batch_size=config["train"]["batch_size"], shuffle=True)
    elif config["name"] in {"fashion"}:
        train_loader = torch.utils.data.DataLoader(
            torchdatasets.FashionMNIST(config["data"]["data_dir"], train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((config["data"]["mean_norm"],), (config["data"]["var_norm"],))
                           ])),
            batch_size=config["train"]["batch_size"], shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            torchdatasets.FashionMNIST(config["data"]["data_dir"], train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((config["data"]["mean_norm"],), (config["data"]["var_norm"],))
                           ])),
            batch_size=config["train"]["batch_size"], shuffle=True)
    else:
        print("Unknown dataset")
        exit()
    
    return train_loader, val_loader
  
 
def load_model(args, config):
    

    
    train_loader, val_loader = load_data(config)
    model_path = os.path.join(args.load, "model")    
    print("loading model", model_path)         
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, ["binmodel"], model_path)
        graph = tf.get_default_graph()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)  
        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("fc_5/add_b:0")                
        #[print(n.name) for n in tf.get_default_graph().as_graph_def().node] 
        
        ################# extract all paramaters 
        nb_layer = len(config["model"]["layers"]) + 1
        #print(nb_layer)
        tf_weights = []
        tf_biases = []
        tf_running_mean  = []             
        tf_running_var   = []
        tf_gamma         = []
        tf_beta          = []
        tf_eps           = []
        tf_runstd        = []
        tf_invstd        = []

        for i in range(nb_layer):
            w =  graph.get_tensor_by_name("fc_{}/W{}:0".format(i,i))
            b =  graph.get_tensor_by_name("fc_{}/b{}:0".format(i,i))
            tf_weights.append(w)
            tf_biases.append(b)
            
            if (i == nb_layer - 1):
                continue
            
            running_mean    = graph.get_tensor_by_name("bn_{}/running_mean{}:0".format(i,i))             
            running_var     = graph.get_tensor_by_name("bn_{}/running_var{}:0".format(i,i))
            gamma     = graph.get_tensor_by_name("bn_{}/gamma{}:0".format(i,i))
            beta     = graph.get_tensor_by_name("bn_{}/beta{}:0".format(i,i))
            #invstd     = graph.get_tensor_by_name("bn_{}/invstd{}:0".format(i,i))
            eps             =  graph.get_tensor_by_name("bn_{}/eps{}:0".format(i,i)) 
                    

            tf_running_mean.append(running_mean)
            tf_running_var.append(running_var)
            tf_gamma.append(gamma)
            tf_beta.append(beta)
            tf_eps.append(eps)
        #############################################
        
        top1 = AverageMeter()        
        for i, (inputs, target) in enumerate(val_loader):
            target = target.view(target.shape[0])
            target = torch.LongTensor(target)
            input = inputs.view(-1, 28*28).clone().cpu().detach().numpy()
            output = sess.run(y, feed_dict={x:input})  
            prec  = accuracy(torch.Tensor(output), target)
            top1.update(prec, inputs.size(0))
            print("batch", i, top1.avg)
            
            ########### simulation #######################
            out =  input
            for j in range(nb_layer):
                #print(tf_weights[j])
                lin = tf.add(tf.matmul(out,  tf.transpose(tf_weights[j])), tf_biases[j])
                if j < nb_layer -1:
                    tf_invstd = 1 / tf.sqrt(tf.add(tf.dtypes.cast(tf_running_var[j], dtype=tf.float32 ), tf.dtypes.cast(tf_eps[j], dtype=tf.float32 )))  
                    bn = tf.add(tf.add(lin, - tf_running_mean[j])*tf_invstd*tf_gamma[j], tf_beta[j])
                    out = tf.sign(bn)
                else:
                     out = lin
            sim_output = sess.run(out)
            #############################################
            
            # compare simulated output and forward run
            assert(np.allclose(np.abs(sim_output), np.abs(output), atol = 0.001, rtol = 0.001))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='BNNs loader')
    parser.add_argument('-l', '--load', default=None, type=str,
                            help='load stored model from a dir')
    args = parser.parse_args()
  
    if not (args.load is None):
        try:
            config = json.load(open(os.path.join(args.load, CONFIG_DEFAULT_FILE_NAME)))
            random.seed(config["manual_seed"])                
        except Exception as e:
            print("Error in reading {} from {}, error {}".format("config", args.load, e))
            exit()        
        load_model(args, config)
    else:
        print("Provide a path to the model")
        exit()
