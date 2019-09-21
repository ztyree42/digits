import yaml

with open('/home/ztyree/projects/digits/digits/params.yaml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    print(data)

data['decomposer']['data']