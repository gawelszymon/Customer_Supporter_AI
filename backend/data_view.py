from flask import Flask, render_template
import pandas as pd

app = Flask(__name__, template_folder='../frontend/templates')


def load_data():
    df = pd.read_csv('../cennik.csv', sep=';', encoding='utf-8', on_bad_lines='skip')

    df.columns = df.columns.str.strip().str.replace(r'\n', '', regex=True)

    df['MAIN_PICTURE'] = df['MAIN_PICTURE'].apply(lambda x: f'<img src="{x}" width="100">' if pd.notnull(x) else '')
    df['PICTURES'] = df['PICTURES'].apply(
        lambda x: '<br>'.join([f'<img src="{img}" width="100">' for img in x.split(':')]) if pd.notnull(x) else '')

    return df.head(100)


@app.route('/')
def index():
    data = load_data()
    return render_template('table.html',tables=[data.to_html(classes='data', header="true", escape=False, index=False)])


if __name__ == '__main__':
    app.run(debug=True, port=8083)
