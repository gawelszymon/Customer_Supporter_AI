with open("base/cennik.csv", 'r') as file:
    content = file.read()

content = content.replace('`', ';')

with open("base/cennik.csv", 'w') as file:
    file.write(content)