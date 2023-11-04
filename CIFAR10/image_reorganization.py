import os

if __name__ == '__main__':
    data_splits = ['train', 'test']

    for split in data_splits:
        classes = os.listdir(split)
        # print(classes)

        combined_path = os.path.join(split, 'images')
        os.mkdir(combined_path)

        for c in classes:
            cpath = os.path.join(split, c)
            print(cpath)
            images = os.listdir(cpath)

            for image in images:
                ipath = os.path.join(cpath, image)
                new_path = os.path.join(combined_path, f'{c}_{image}')
                os.rename(ipath, new_path)
