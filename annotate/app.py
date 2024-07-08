from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

all_points = []
image_filename = 'frame.jpg'  # Define your image filename here

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html', image_filename=image_filename)

# Route to handle mouse click coordinates
@app.route('/log_click', methods=['POST'])
def log_click():
    data = request.json
    x = data.get('x')
    y = data.get('y')
    print(f"({x}, {y})")  # You can log this to a file or database
    all_points.append((x, y))
    print(all_points)
    return jsonify(success=True)

if __name__ == '__main__':
    app.run(debug=True)

