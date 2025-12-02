import lmdb

env = lmdb.open('/home/yys/ai_hub_package/train_data/MJ', readonly = True)

with env.begin () as txn:

    path_key = 'images'.encode()
    path_value = txn.get(path_key).decode()

    img_key = "name".encode()
    img_value = txn.get(img_key).decode()
