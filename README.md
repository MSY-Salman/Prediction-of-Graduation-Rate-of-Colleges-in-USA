# College Data Analysis and Visualization

This is a Streamlit application designed to analyze and visualize data related to colleges in the USA. The dataset contains various attributes for colleges, such as graduation rates, faculty count, student enrollments, and more. The application provides interactive data visualizations to explore these relationships and gain insights into factors affecting graduation rates.
Can be accessed at: https://prediction-of-graduation-rate-of-colleges-in-usa.streamlit.app/
## Features:
- **Graduation Rate Distribution**: Visualizes the distribution of graduation rates across all colleges.
- **PhD Faculty vs Graduation Rate**: A scatterplot comparing the number of PhD faculty with the graduation rate, split by private and public colleges.
- **Tuition Fees vs Graduation Rate**: A scatterplot comparing out-of-state tuition fees with graduation rates.
- **Correlation Heatmap**: A heatmap to explore correlations between various numerical variables.
- **Graduation Rate vs Top 10 Percent**: A scatterplot comparing the percentage of students from the top 10% of their high school class with graduation rates.

## Demo

You can run the app locally by following the steps below.

## Prerequisites

Before running this app locally, ensure you have the following installed:

- Python 3.x
- Required Python libraries (see below)

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/college-data-visualization.git
   ```

2. Navigate to the project directory:

   ```bash
   cd college-data-visualization
   ```

3. Install the necessary dependencies using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

After installing the dependencies, run the application using the following command:

```bash
streamlit run app.py
```

This will open the application in your default web browser.

## File Structure

```
college-data-visualization/
├── app.py            # Main Streamlit application
├── College.csv       # Dataset containing college information
├── requirements.txt  # List of dependencies
└── README.md         # Project README
```

## Dataset

The dataset used in this application is `College.csv`, which contains data on various colleges in the USA, including:

- **Colleges**: College names
- **Private**: Whether the college is private (Yes/No)
- **Apps**: Number of applications
- **Accept**: Number of accepted applicants
- **Enroll**: Number of enrolled students
- **Grad.Rate**: Graduation rate of the college
- **PhD**: Percentage of faculty with a PhD

## Dependencies

This application uses the following Python libraries:

- `pandas`: For data manipulation and analysis.
- `matplotlib`: For creating static, animated, and interactive visualizations.
- `seaborn`: For statistical data visualization.
- `streamlit`: For creating the interactive web application.
- `numpy`: For numerical operations.

You can install the required libraries with the following command:

```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! If you have suggestions or improvements for the project, feel free to fork the repository and submit a pull request. Please follow the coding guidelines and include tests where applicable.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Make sure to replace `yourusername` in the clone URL with your actual GitHub username. If you have a live demo link, you can update the placeholder for that as well. The `requirements.txt` file should include all the dependencies used in the app, like this:

```
pandas
matplotlib
seaborn
streamlit
numpy
```
