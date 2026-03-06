

l0 = [32,32,32]
l1 = [50,50,50]
l2 = [100,100,100]
l3 = [200,200,200]
l4 = [400,400,400]
l5 = [800,800,800]
l6 = [256,256,256]
l7 = [80,80,80]
l8 = [300,300,300]


models = [l6,l2,l6,l2,l7,l7,l8]

model_names = []
for model in models:
    length = len(model)
    name = model[0]
    name_str = "NN" + str(name) + "x"+str(length)

    model_names.append(name_str)

#model_names = ['NN4096x3', 'NN64x3', 'NN2048x3', 'NN128x3', 'NN1024x3', 'NN256x3', 'NN512x3']