# Import necessary libraries from shiny and plotly for building the interactive dashboard
from shiny import App, ui, render, reactive
import plotly.graph_objects as go  # Used for creating plotly figures, especially for adding traces like trend lines
import plotly.express as px  # Used for quick creation of plotly charts like scatter plots and histograms
import numpy as np  # NumPy for generating random data for different distributions
import pandas as pd  # Pandas for data manipulation and storage
from htmltools import HTML  # For rendering HTML inside the Shiny UI

# Define the user interface (UI) for the ShinyLive app
app_ui = ui.page_fluid(
    # Title of the dashboard
    ui.h2("Interactive Data Visualization Dashboard"),
    
    # Layout: Sidebar on the left, content on the right
    ui.layout_sidebar(
        # Sidebar elements (inputs for user interaction)
        ui.sidebar(
            # Slider for selecting the number of data points
            ui.input_slider("n", "Number of points", min=10, max=500, value=100),
            # Dropdown for selecting the type of data distribution
            ui.input_select("distribution", "Distribution", 
                            choices=["Normal", "Uniform", "Exponential"]),
            # Numeric input to specify the number of bins for the histogram
            ui.input_numeric("bins", "Number of bins (Histogram)", value=20, min=5, max=100),
            # Checkbox to toggle displaying a trend line on the scatter plot
            ui.input_checkbox("show_trend", "Show trend line", value=False),
        ),
        # Content layout: split into two columns for the scatter plot and histogram
        ui.row(
            ui.column(6, ui.h3("Scatter Plot"), ui.output_ui("scatterplot")),  # Left column for scatter plot
            ui.column(6, ui.h3("Histogram"), ui.output_ui("histogram"))  # Right column for histogram
        ),
        # Row for displaying the first 10 rows of the generated data in a table
        ui.row(
            ui.column(12, ui.h3("Data Table"), ui.output_table("data_table"))
        )
    )
)

# Define the server logic for the ShinyLive app
def server(input, output, session):
    # Reactive function to generate data based on user inputs (number of points and distribution type)
    @reactive.Calc
    def generate_data():
        n = input.n()  # Get the number of points from the input slider
        dist = input.distribution()  # Get the selected distribution type from the dropdown
        if dist == "Normal":
            # Generate normally distributed data for x and y
            x = np.random.randn(n)
            y = np.random.randn(n)
        elif dist == "Uniform":
            # Generate uniformly distributed data for x and y
            x = np.random.uniform(-3, 3, n)
            y = np.random.uniform(-3, 3, n)
        else:  # Exponential distribution
            # Generate exponentially distributed data for x and y
            x = np.random.exponential(1, n)
            y = np.random.exponential(1, n)
        # Return the data as a Pandas DataFrame
        return pd.DataFrame({'x': x, 'y': y})

    # Render the scatter plot dynamically based on the generated data
    @output
    @render.ui
    def scatterplot():
        df = generate_data()  # Get the generated data
        # Create a scatter plot using Plotly Express
        fig = px.scatter(df, x='x', y='y', title="Interactive Scatter Plot")
        
        # Add a trend line if the checkbox is checked
        if input.show_trend():
            fig.add_trace(go.Scatter(x=df['x'], y=df['y'].rolling(window=20).mean(),
                                     mode='lines', name='Trend', line=dict(color='red')))
        
        # Update the plot title with the selected distribution type
        fig.update_layout(title=f"Scatter Plot ({input.distribution()} Distribution)")
        # Convert the plotly figure to HTML and return it to be rendered in the UI
        plot_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
        return ui.div(HTML(plot_html))  # Render the scatter plot in the UI

    # Render the histogram dynamically based on the generated data
    @output
    @render.ui
    def histogram():
        df = generate_data()  # Get the generated data
        # Create a histogram using Plotly Express
        fig = px.histogram(df, x='x', nbins=input.bins(), title="Histogram of X values")
        # Update the plot title with the selected distribution type
        fig.update_layout(title=f"Histogram ({input.distribution()} Distribution)")
        # Convert the plotly figure to HTML and return it to be rendered in the UI
        plot_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
        return ui.div(HTML(plot_html))  # Render the histogram in the UI

    # Render a table that shows the first 10 rows of the generated data
    @output
    @render.table
    def data_table():
        # Display the first 10 rows of the data table
        return generate_data().head(10)

# Create the ShinyLive app by passing the UI and server logic
app = App(app_ui, server)