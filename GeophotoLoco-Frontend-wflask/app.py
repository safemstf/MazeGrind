from flask import Flask, render_template

app = Flask(__name__, static_folder='static')

@app.route('/input')
def input():
    return render_template('input.html')

@app.route('/output')
def output():
    return render_template('output.html')

if __name__ == '__main__':
    app.run(debug=True)