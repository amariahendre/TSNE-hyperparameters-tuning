# T-SNE Visualization Tool

The T-SNE Visualization Tool is a specialized utility designed to help you fine-tune the hyperparameters of the t-distributed Stochastic Neighbor Embedding (t-SNE) algorithm for optimal visualization of high-dimensional data in a two-dimensional space.

## Requirements:
- numpy
- streamlit
- scikit-learn
- matplotlib

## Features:
1. **Data Upload**: Users can upload their own file containing their dataset through the Streamlit sidebar.
2. **Data Transformation**: The tool applies a transformation to the data by taking the logarithm of each value plus one (\( \log_2(x+1) \)). This step helps normalize and preprocess the data for visualization.
3. **PCA (Principal Component Analysis)**: Users can perform Principal Component Analysis on the transformed data by adjusting the number of PCA components.
4. **T-SNE Hyperparameter Tuning**: Users can configure various t-SNE hyperparameters, such as perplexity, learning rate, early exaggeration, initialization method, and the number of iterations.
5. **Visualization**: A scatter plot is generated in two dimensions based on the chosen t-SNE hyperparameters. Data points are represented as dots in the scatter plot.

## Author:
Made with love by [Ana Maria Hendre](https://www.linkedin.com/in/anamariahendre/)

## License:
The code is open-source and available for modification and use. Kindly provide proper attribution to the author if you decide to use it.
