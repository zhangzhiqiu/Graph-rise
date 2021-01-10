import os
import numpy as np
import tensorflow as tf
import cv2 as cv
from tensorflow import keras
import scipy.io as scio
import csv
from string import digits

import random


def one_hot(labels):
    labels_metric = np.zeros([len(labels),labels.max()+1])
    for i in range(len(labels)):
        labels_metric[i,labels[i]] = 1
    return labels_metric


def load_image(file_path, attr, train = 'train'):
    class_dir = os.listdir(os.path.join(file_path,train))
    class_num = len(class_dir)
    train_labels = []
    train_images = []
    train_attr = []
    class_dict = dict()
    class_count = 0
    for class_ in class_dir:
        img_dir = os.listdir(os.path.join(file_path,train,class_))
        for image_ in img_dir:
            image_name = os.path.join(os.path.join(file_path,train,class_, image_))
            img = cv.imread(image_name)
            img = cv.resize(img,(224,224))
            img = img.astype('float32') / 127.5 - 1
            train_images.append(img)
            train_labels.append(class_count)
            train_attr.append(attr[image_.split('.')[0]])
            print(image_)
            print(attr[image_.split('.')[0]])
            #train_labels = one_hot(np.array(train_labels))
        class_dict[class_] = class_count
        class_count += 1

    train_images = np.array(train_images).reshape(-1, 224, 224, 3)
    permutation = np.random.permutation(train_images.shape[0])
    train_images = train_images[permutation, :, :, :]


    train_labels = np.array(train_labels)
    train_labels = train_labels[permutation]
    train_labels = keras.utils.to_categorical(train_labels,train_labels.max()+1)

    train_attr = np.array(train_attr)
    train_attr = train_attr[permutation]

    return train_images,train_labels,class_dict,train_attr


'''
def load_test(file_path, class_dict):
    test_file = open(os.path.join(file_path,'val','val_annotations.txt'),'r')
    test_images = []
    test_labels = []
    for line in test_file:
        name = line.split()
        image_name = os.path.join(file_path,'val','images',name[0])
        img = cv.imread(image_name)
        img = cv.resize(img, (224, 224))
        img = img.astype('float32') / 127.5 - 1
        test_images.append(img)
        test_labels.append(class_dict[name[1]])

    test_labels = np.array(test_labels)
    test_labels = keras.utils.to_categorical(test_labels, test_labels.max()+1)
    test_images = np.array(test_images).reshape(-1, 224, 224, 3)

    return test_images, test_labels

'''
def load_test(file_path, class_dict):
    #test_file = open(os.path.join(file_path,'val','val_annotations.txt'),'r')
    test_file = os.listdir(os.path.join(file_path,'test'))
    test_images = []
    test_labels = []
    for line in test_file:
        file_dir = os.listdir(os.path.join(file_path,'test',line))
        for test_name in file_dir[:80]:
            image_name = os.path.join(file_path,'test',line,test_name)
            img = cv.imread(image_name)
            img = cv.resize(img, (224, 224))
            img = img.astype('float32') / 127.5 - 1
            test_images.append(img)
            test_labels.append(class_dict[line])

    test_labels = np.array(test_labels)
    test_labels = keras.utils.to_categorical(test_labels, test_labels.max()+1)
    test_images = np.array(test_images).reshape(-1, 224, 224, 3)

    return test_images, test_labels

def load_graph(graph_file):
    data = scio.loadmat(graph_file)
    a = data['attrann']
    c = a[0][0]
    attr = c[2]
    attr = attr + np.ones(np.shape(attr))
    name = c[0]
    attr_dict = dict()
    count = 0
    for i in name:
        s = i[0][0]
        attr_dict[s] = attr[count]
        count += 1
    return attr_dict


def load_bird(file_path='./data/CUB_200_2011'):
    def load_dir(file_path='./data/CUB_200_2011'):
        name_list = os.path.join(file_path,'images.txt')
        dir_dict = dict()
        for line in open(name_list):
            image_id, dir = line.split()
            dir_dict[image_id] = dir
        return dir_dict

    def load_classes(file_path='./data/CUB_200_2011'):
        name_list = os.path.join(file_path,'image_class_labels.txt')
        class_dict = dict()
        for line in open(name_list):
            image_id, class_id = line.split()
            class_dict[image_id] = class_id
        return class_dict

    def load_train_test(file_path='./data/CUB_200_2011'):
        name_list = os.path.join(file_path,'train_test_split.txt')
        train_test_dict = dict()
        cout_train = 0
        cout_test = 0
        for line in open(name_list):
            image_id, train_test = line.split()
            train_test_dict[image_id] = train_test
            '''if train_test == '1':
                cout_train +=1
            else:
                cout_test += 1
            print('train',cout_train,'test',cout_test,train_test)'''
        return train_test_dict

    def load_boxes(file_path='./data/CUB_200_2011'):
        name_list = os.path.join(file_path,'bounding_boxes.txt')
        box_dict = dict()
        for line in open(name_list):
            image_id, x, y, width, height = line.split()
            box_dict[image_id] = np.array([float(x),float(y),float(width),float(height)])
        return box_dict
    def load_attr(file_path='./data/CUB_200_2011'):
        name_list = os.path.join(file_path,'attributes','image_attribute_labels.txt')
        attr_dict = dict()
        for line in open(name_list):
            image_id, attr_id, is_, attr_ = line.split()[0:4]

            if int(attr_id) == 1:
                attr_dict[image_id] = []
            attr_dict[image_id].append(int(attr_)*int(is_))
        return attr_dict

    def slim_graph():
        slim_graph = []
        with open('./data/graph.csv','r') as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                w = row[2]
                if not w == '0':
                    slim_graph.append([row[0],row[1],row[2]])

                else:
                    print(row[0])
        print('finish_1')
        with open('./data/graph1.csv', 'w') as f:
            f_csv = csv.writer(f)
            for row in slim_graph:
                f_csv.writerow(row)
        print('finish_all')


    def build_graph(attr,dir_dict,class_dict):
        from_to_w = []
        weight = []
        inner_count = 0
        outer_count = 0
        outer_w_sum = 0
        inner_w_sum = 0
        with open('./data/graph.csv','w') as f:
            f_csv = csv.writer(f)
            for i in range(11788):
                print(i)
                for j in range(11788):
                    if not i == j:
                        w = np.sum(np.array(attr[str(i+1)]) * np.array(attr[str(j+1)]))
                        from_to_w.append([dir_dict[str(i+1)],dir_dict[str(j+1)],w])
                        weight.append(w)

                        if class_dict[str(i+1)] == class_dict[str(j+1)]:
                            inner_count += 1
                            inner_w_sum += w

                        else:
                            outer_count += 1
                            outer_w_sum += w

                        f_csv.writerow([dir_dict[str(i+1)],dir_dict[str(j+1)],w])
                print('out:',outer_w_sum/outer_count,inner_w_sum/inner_count)
        return from_to_w,np.array(weight)

    #slim_graph()
    dir_dict = load_dir(file_path)
    boxes = load_boxes(file_path='./data/CUB_200_2011')
    class_dict = load_classes(file_path)
    attrs = load_attr(file_path)
    train_test = load_train_test(file_path)

    #from_to_w,weight = build_graph(attrs, dir_dict,class_dict)
    #print(from_to_w,weight.max())

    train_images = []
    train_labels = []
    train_attr = []
    test_images = []
    test_labels = []
    test_attr = []


    for id in range(1,11789):
        str_id = str(id)
        image = np.array(cv.imread(os.path.join(file_path,'images',dir_dict[str_id])))
        box = boxes[str_id]
        image = image[int(box[1]):int(box[1])+int(box[3]),int(box[0]):int(box[0])+int(box[2]),:]
        image = cv.resize(image,(224,224))
        image = image.astype('float32') / 127.5 - 1
        label = int(class_dict[str_id])-1
        attr = attrs[str_id]
        if train_test[str_id] == '1' or (not id%5 == 0):
            train_images.append(image)
            train_labels.append(label)
            train_attr.append(attr)
        else:
            test_images.append(image)
            test_labels.append(label)
            test_attr.append(attr)

    train_images = np.array(train_images).reshape(-1, 224, 224, 3)
    permutation = np.random.permutation(train_images.shape[0])
    train_images = train_images[permutation, :, :, :]

    train_labels = np.array(train_labels)
    train_labels = train_labels[permutation]
    train_labels = keras.utils.to_categorical(train_labels, train_labels.max() + 1)

    train_attr = np.array(train_attr)
    train_attr = train_attr[permutation]

    test_images = np.array(test_images).reshape(-1, 224, 224, 3)

    test_labels = np.array(test_labels)
    test_labels = keras.utils.to_categorical(test_labels, test_labels.max() + 1)

    test_attr = np.array(test_attr)

    print(np.shape(train_images),np.shape(train_labels),np.shape(train_attr))
    print(np.shape(test_images), np.shape(test_labels), np.shape(test_attr))
    return train_images, train_labels, train_attr, test_images, test_labels, test_attr

def load_fossil(file_path,train=1,image_size=448):
    if train==1:
        dir = 'train_images'
    else:
        dir = 'test_images'
    train_images = []
    train_labels = []
    train_imagesv = []
    train_labelsv = []
    genus_set    = []
    graph = []
    genus = dict()

    class_list = os.listdir(os.path.join(file_path,dir))
    for sub_class in class_list:
        genus = sub_class.split()[0]
        genus = genus.translate(str.maketrans('', '', digits))
        genus_set.append(genus)
    genus_set = list(set(genus_set))
    genus_dict = dict()
    specise_dict = dict()
    class_dict = dict()
    genus_num = dict()
    genus_count = 0
    for genus_name in genus_set:
        genus_num[genus_name] = genus_count
        genus_count += 1
        genus_dict[genus_name] = []
        class_count = 0
        for sub_class in class_list:
            class_dict[sub_class] = class_count
            class_count += 1
            if genus_name in sub_class:
                genus_dict[genus_name].append(sub_class)
                specise_dict[sub_class] = genus_name


    #np.save('save/genus.npy', genus_num)
    genus_num = np.load('save/genus.npy',allow_pickle=True).item()




    select_class = class_list

    if train == 2:
        class_list = select_class
    f = open('save/file_num.txt','w')
    for class_ in class_list:
        f.writelines(class_+' '+specise_dict[class_]+' '+str(genus_num[specise_dict[class_]])+'\n')
        print(class_,specise_dict[class_],genus_num[specise_dict[class_]])
        image_list = os.listdir(os.path.join(file_path, dir, class_))
        for image_dir in image_list:
            image = cv.imread(os.path.join(file_path, dir, class_,image_dir))
            #print(os.path.join(file_path, dir, class_,image_dir))
            image= cv.resize(image,(image_size,image_size))
            image = image.astype('float32') / 127.5 - 1
            train_images.append(image)
            train_labels.append(genus_num[specise_dict[class_]])


            same_dir = image_list
            random_choice_img_num = random.randint(0, np.shape(same_dir)[0])
            random_choice_img = same_dir[random_choice_img_num - 1]
            imagev = cv.imread(os.path.join(file_path, dir, class_, random_choice_img))
            imagev = cv.resize(imagev, (image_size, image_size))
            imagev = imagev.astype('float32') / 127.5 - 1
            train_imagesv.append(imagev)
            train_labelsv.append(genus_num[specise_dict[class_]])

            if class_ in select_class:
                edge = float(random.randint(7500,10000))/10000
            else:
                edge = 0
            # print('same_pair', os.path.join(file_path, dir, class_, random_choice_img))
            graph.append(edge)



    print('Now..................shuft1')

    train_images = np.array(train_images).reshape(-1, image_size, image_size, 3)
    permutation = np.random.permutation(train_images.shape[0])
    train_images = train_images[permutation, :, :, :]
    print('Now..................shuft2')

    train_labels = np.array(train_labels)
    train_labels = train_labels[permutation]
    train_labels = keras.utils.to_categorical(train_labels, train_labels.max()+1)
    print('Now..................shuft3')

    '''train_imagesv = np.array(train_imagesv).reshape(-1, image_size, image_size, 3)
    train_imagesv = train_imagesv[permutation, :, :, :]
    print('Now..................shuft4')

    train_labelsv = np.array(train_labelsv)
    train_labelsv = train_labelsv[permutation]
    train_labelsv = keras.utils.to_categorical(train_labelsv, train_labelsv.max()+1)
    print('Now..................shuft5')'''
    train_imagesv = train_images
    train_labelsv = train_labels

    graph = np.array(graph)
    graph = graph[permutation]
    #train_attr = np.array(train_labels)
    #train_attr = train_attr[permutation]

    print(np.shape(train_labels))
    return train_images,train_labels,train_imagesv,train_labelsv, graph


def load_HC_fossil(file_path,train=1,image_size=448):
    if train==1:
        dir = 'train_images'
    else:
        dir = 'test_images'
    train_images = []
    train_labels_SP = []
    train_labels_GE = []
    train_labels_HI = []
    genus_set    = []

    mean = np.array([0.948078, 0.93855226, 0.9332005])
    var = np.array([0.14589554, 0.17054074, 0.18254866])

    class_list = os.listdir(os.path.join(file_path,dir))
    for sub_class in class_list:
        genus = sub_class.split()[0]
        genus = genus.translate(str.maketrans('', '', digits))
        genus_set.append(genus)
    genus_set = list(set(genus_set))

    genus_dict = dict()
    specise_dict = dict()
    class_dict = dict()
    genus_num = dict()
    sp_ge_dict = dict()

    genus_count = 0
    for genus_name in genus_set:
        genus_num[genus_name] = genus_count
        genus_count += 1
        genus_dict[genus_name] = []
        class_count = 0
        for sub_class in class_list:
            class_dict[sub_class] = class_count
            class_count += 1
            if genus_name in sub_class:
                genus_dict[genus_name].append(sub_class)
                specise_dict[sub_class] = genus_name

    #np.save('save/genus.npy', genus_num)
    genus_num = np.load('save/genus.npy',allow_pickle=True).item()

    for class_ in class_list:
        image_list = os.listdir(os.path.join(file_path, dir, class_))
        print(len(image_list))
        for image_dir in image_list:
            image = cv.imread(os.path.join(file_path, dir, class_,image_dir))
            image= cv.resize(image,(image_size,image_size))
            image = ((image.astype('float32') / 127.5 - mean)/var).astype('float32')

            train_images.append(image)
            train_labels_SP.append(class_dict[class_])
            train_labels_GE.append(genus_num[specise_dict[class_]])
            sp_ge_dict[class_dict[class_]] = genus_num[specise_dict[class_]]


    print('Now..................shuft1')
    train_images = np.array(train_images).reshape(-1, image_size, image_size, 3)
    '''permutation = np.random.permutation(train_images.shape[0])
    train_images = train_images[permutation, :, :, :]'''

    print('Now..................shuft2')
    train_labels_SP = np.array(train_labels_SP)
    #train_labels_SP = train_labels_SP[permutation]
    train_labels_SP = keras.utils.to_categorical(train_labels_SP, train_labels_SP.max()+1)

    print('Now..................shuft3')
    train_labels_GE = np.array(train_labels_GE)
    #train_labels_GE = train_labels_GE[permutation]
    train_labels_GE = keras.utils.to_categorical(train_labels_GE, train_labels_GE.max() + 1)


    return train_images,train_labels_SP,train_labels_GE,sp_ge_dict


'''count_all = 0
count_real = 0
for i in range(len(train_images)):
    label_SP = parallel_model(train_images[i:i+1])
    count_all += 1
    if sp_ge[np.array(label_SP).argmax()] == train_labels_GE[i].argmax():
        count_real += 1
    print('GE_REAL=', float(count_real) / float(count_all))'''
'''parallel_model.fit({'input1': train_images},
                       {'out_GE': train_labels_GE},
                       epochs=args.epoch, verbose=1, batch_size=args.batch_size,
                       callbacks=[checkpoint, reduce_lr])'''
'''parallel_model.fit({'input1': train_images},
                           {'out_SP': train_labels_SP},
                           epochs=args.epoch, verbose=1, batch_size=args.batch_size,
                           callbacks=[checkpoint, reduce_lr])'''
'''parallel_model.fit({'input1': train_images},
                           {'out_GE': train_labels_GE, 'out_SP': train_labels_SP},
                           epochs=args.epoch, verbose=1, batch_size=args.batch_size,
                           callbacks=[checkpoint, reduce_lr])'''
#parallel_model = HC_T_compile(args, [args.image_size, args.image_size, 3], np.shape(train_labels_GE)[1],np.shape(train_labels_SP)[1])
    #parallel_model = HC_GE_compile(args, [args.image_size, args.image_size, 3], np.shape(train_labels_GE)[1],np.shape(train_labels_SP)[1])
    #parallel_model = HC_SP_compile(args, [args.image_size, args.image_size, 3], np.shape(train_labels_GE)[1],np.shape(train_labels_SP)[1])