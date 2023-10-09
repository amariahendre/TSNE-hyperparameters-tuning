import numpy as np
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

st.set_page_config(page_title='T-SNE Vizualization')


st.title('T-SNE Visualization Tool')
st.write('The T-SNE Visualization Tool is a specialized utility designed to help you fine-tune the hyperparameters of the t-distributed Stochastic Neighbor Embedding (t-SNE) algorithm for optimal visualization of high-dimensional data in a two-dimensional space.')

uploaded_file = st.sidebar.file_uploader("Upload a .npy file", type="npy")

text = ("""
Data upload: In the sidebar, you can upload your own .npy file containing your dataset. 

Data transformation: The tool applies a transformation to the data by taking the logarithm of each value plus one (log2(x+1)). This step helps normalize and preprocess the data for visualization.

PCA (Principal Component Analysis): You can perform Principal Component Analysis on the transformed data by adjusting the number of PCA components. PCA reduces the dimensionality of the data while preserving its most important features.

T-SNE hyperparameter tuning: Configure the following t-SNE hyperparameters to optimize your visualization.
- Perplexity: Adjust the perplexity parameter, which influences the balance between preserving local and global structures in the data. Perplexity can be considered as a smooth measure of the effective number of neighbors. A larger dataset usually requires a larger perplexity. Consider selecting a value between 5 and 100.
- Learning Rate: Set the learning rate for the t-SNE algorithm, which affects the convergence speed. The learning rate can be a critical parameter. If the learning rate is too high, the data may look like a 'ball' with any point approximately equidistant from its neighbor. A typical value to start with is 200.
- Early Exaggeration: Modify the early exaggeration parameter to control the spacing of clusters in the visualization.  Larger values ensure that natural clusters are tighter.
- Initialization: Choose between random initialization or PCA initialization for t-SNE.
- Number of Iterations: Set the number of iterations for t-SNE optimization.

Visualization with hyperparameters: The T-SNE Visualization Tool generates a scatter plot in two dimensions based on your chosen t-SNE hyperparameters. Data points are represented as dots in the scatter plot.
The title of the plot includes the selected t-SNE hyperparameters for reference.
""" )

# If a new file is uploaded, overwrite the default data
if uploaded_file is not None:
    with st.expander("How to use this tool"):
        st.write(text)
    st.write("---")

    # Load data based on file format
    if uploaded_file.name.endswith('.npy'):
        data = np.load(uploaded_file)
    elif uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file).values
    elif uploaded_file.name.endswith('.txt'):
        data = np.loadtxt(uploaded_file)
        
    st.sidebar.write('Data shape:', data.shape)
    st.sidebar.write("---")


    # Add checkbox for data transformation
    apply_transformation = st.sidebar.checkbox('Apply log2(x+1) transformation to data', value=True)

    # Conditionally transform the data using log2(x+1)
    if apply_transformation:
        data = np.log2(data + 1)

    # PCA
    st.sidebar.subheader('PCA Parameters')
    n_components = st.sidebar.slider('Number of PCA components', 1, min(data.shape[0], data.shape[1]), 50)
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(data)
    # st.sidebar.write("---")

    # T-SNE parameters
    st.sidebar.subheader('T-SNE Parameters')
    data_choice = st.sidebar.selectbox('Choose data for t-SNE', ['Original or Transformed Data', 'PCA Data'])
    perplexity = st.sidebar.slider('Perplexity', 2, 100, 30)
    learning_rate = st.sidebar.slider('Learning Rate', 10, 1000, 200)
    early_exaggeration = st.sidebar.slider('Early Exaggeration', 1, 50, 12)
    init = st.sidebar.selectbox('Initialization', ['random', 'pca'])
    n_iter = st.sidebar.slider('Number of Iterations', 250, 5000, 1000)

    st.sidebar.write("---")
    st.sidebar.markdown(
                '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://www.linkedin.com/in/anamariahendre/">@anamariahendre</a></h6>',
                unsafe_allow_html=True,
            )

    if data_choice == 'PCA Data':
        input_data = pca_data
    else:
        input_data = transformed_data

    tsne = TSNE(perplexity=perplexity, learning_rate=learning_rate, early_exaggeration=early_exaggeration,
                init=init, n_iter=n_iter, random_state=42)

    tsne_data = tsne.fit_transform(input_data)

    # Visualization with Matplotlib
    fig, ax = plt.subplots()
    ax.scatter(tsne_data[:, 0], tsne_data[:, 1], color='#ff4b4b', alpha=0.5)
    ax.set_xlim([tsne_data[:, 0].min() - 5, tsne_data[:, 0].max() + 5])
    ax.set_ylim([tsne_data[:, 1].min() - 5, tsne_data[:, 1].max() + 5])
    plt.gca().set_aspect('equal', adjustable='box')

    # Set the title with all the t-SNE parameters
    title_str = (
        f"T-SNE Visualization\n"
        f"Data Choice: {data_choice}\n"
        f"Perplexity: {perplexity}\n"
        f"Learning Rate: {learning_rate}\n"
        f"Early Exaggeration: {early_exaggeration}\n"
        f"Initialization: {init}\n"
        f"Number of Iterations: {n_iter}\n"
    )

    ax.set_title(title_str, fontsize=9)
    plt.axis("equal")

    st.pyplot(fig)

else:
   st.header("How to use this app:")
   st.write(text)

