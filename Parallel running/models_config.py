

l0 = [64,64,64]
l1 = [128,128,128]
l2 = [256,256,256]
l3 = [512,512,512]
l4 = [1024,1024,1024]
l5 = [2048,2048,2048]
l6 = [4096,4096,4096]


models = [l6,l0,l5,l1,l4,l2,l3]

model_names = []
for model in models:
    length = len(model)
    name = model[0]
    name_str = "NN" + str(name) + "x"+str(length)

    model_names.append(name_str)

#model_names = ['NN4096x3', 'NN64x3', 'NN2048x3', 'NN128x3', 'NN1024x3', 'NN256x3', 'NN512x3']