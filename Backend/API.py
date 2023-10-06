from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)


@app.route('/start_game', methods=['POST'])
def start_game():
    import Test_Optimal_actions
    try:
        data = request.get_json(force=True)
        print(data['maze_number'])
        result = Test_Optimal_actions.generate_grid_from_csv(
            filename="optimal_actions.csv", maze_number=data['maze_number'])
        return jsonify({"message": result}), 200
    except Exception as e:
        error_message = str(e)
        result = {"error": error_message}
        return jsonify(result), 500


if __name__ == '__main__':
    app.run(debug=True)
