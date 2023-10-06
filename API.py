from flask import Flask, request, jsonify
app = Flask(__name__)

# POST endpoint

# Get the data from the POST request.
# data = request.get_json(force=True)


@app.route('/start_game', methods=['POST'])
def start_game():
    import Test_Optimal_actions
    try:
        data = request.get_json(force=True)
        # print(data)
        result = Test_Optimal_actions.generate_grid_from_csv(
            filename="optimal_actions.csv", maze_number=data['maze_number'])
        result_message = f"main executed with result: {result}"
        # print(result_message)
        return jsonify({"message": result_message}), 200
    except Exception as e:
        error_message = str(e)
        result = {"error": error_message}
        return jsonify(result), 500


if __name__ == '__main__':
    app.run(debug=True)
