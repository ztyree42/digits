import yaml

with open('/home/ubuntu/projects/digits/digits/params.yaml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    print(data)

data['decomposer']['data']