<p align="center">
  <img src="https://github.com/RicPiz/wine-combinator/blob/main/winecombinator.png" alt="Logo"/>
</p>

![Wine Combinator Logo](winecombinator.png)

An interactive Streamlit application for discovering and recommending Italian wines based on color, price range, and food pairings. Data sourced from Vivino, focusing exclusively on Italian wines.

## Features

- **Wine Recommendations**: Filter wines by color (red/white), price range, and food pairings. Results are ranked by average rating, number of reviews, and proximity to your selected price range.
- **Visual Cards**: Each recommendation displays a bottle image, key details (rating, reviews, price), tasting notes, and structure metrics (light-structured, dry-sweet, flat-acidic).
- **Food Pairing**: Select from various food types (e.g., Beef, Pasta, Fish) to find matching wines.
- **Wineries Map**: Interactive map of Italy showing winery locations. Click on regions to view associated wines and details.
- **Downloadable Results**: Export your recommendations as a CSV file.
- **Responsive Design**: Grid layout for recommendations, custom styling for a polished look.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/wine-combinator.git
   cd wine-combinator
   ```

2. Create a virtual environment (recommended) and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser at `http://localhost:8501`.

3. Use the sidebar filters to select wine color, food pairings, price range, and number of recommendations.

4. Switch to "Wineries Map" mode to explore winery locations interactively.

## Data and Images

- **Wine Data**: Sourced from `red_wines_clean.csv` and `white_wines_clean.csv` (cleaned data from Vivino).
- **Images**: Bottle images are in `img_all/` directory, matched by normalized wine name and year.
- **Map Data**: GeoJSON and shapefiles in `Wine_Map/` for Italian regions and winery coordinates.
- **Disclaimer**: All data and images are obtained from [Vivino](https://www.vivino.com). This project uses only Italian wine information for educational purposes.

If an image isn't found, a placeholder is displayed.

## Project Structure

- `app.py`: Main Streamlit application.
- `red_wines_clean.csv`, `white_wines_clean.csv`: Wine datasets.
- `img_all/`: Directory containing bottle images.
- `Wine_Map/`: Contains geo data for the map feature.
- `requirements.txt`: Python dependencies.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

1. Fork the repo.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Built with ❤️ and 🍷 by Riccardo Pizzuti*










