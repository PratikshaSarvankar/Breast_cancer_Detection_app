import os
import gdown
from tensorflow.keras.models import load_model
import streamlit as st

# Define the model file path
model_path = 'attention_unet_classification_model.keras'

# Function to download the model
def download_model(file_id, model_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    try:
        gdown.download(url, model_path, quiet=False)
        st.success("Model downloaded successfully.")
    except Exception as e:
        st.error(f"Error downloading the model: {str(e)}")

# Check if the model file already exists
if not os.path.exists(model_path):
    # If not, download it
    st.info("Model not found. Attempting to download...")
    # Use the actual file ID from your Google Drive link
    file_id = '1CFi_ctM2KIVxkJzqHapJBoplJrDEVssr'
    download_model(file_id, model_path)
else:
    st.success("Model already downloaded.")

# Load the model
try:
    model = load_model(model_path)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")




# (Add your existing Streamlit app code below)

def first_page():
    # Custom CSS for styles
    st.markdown(
        """
        <style>
        .reportview-container,.main,.stApp{
            background-color: #FADADD!important;  /* Soothing blue-gray background */
        }
        .title {
            text-align: center;
            font-size: 48px;
            color: #003366;
            font-family: 'Arial', sans-serif;
            margin: 20px;
        }
        .message {
            font-size: 24px;  /* Increased font size for better visibility */
            color: #FFFFFF;  /* White color for better contrast */
            font-family: 'Georgia', serif;
            line-height: 1.5;
            background-color: rgba(0, 0, 0, 0.5);  /* Semi-transparent background */
            border-radius: 10px;  /* Rounded corners for the background */
            display: flex;
            align-items: center;  /* Center content vertically */
            justify-content: center;  /* Center content horizontally */
            text-align: center;  /* Center the text */
            padding: 10px;  /* Reduced padding for closer alignment */
            height: 400px;  /* Set a fixed height for the message container */
        }
        .container {
            display: flex;  /* Use flexbox to arrange items */
            justify-content: center;  /* Center horizontally */
            align-items: center;  /* Center vertically */
            margin-top: 50px;
        }
        .image-container {
            flex: 1;  /* Allow image to take up equal space */
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .message-container {
            flex: 1;  /* Allow message to take up equal space */
            margin-left: 10px;  /* Reduce margin to bring the message closer */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title
    st.markdown("<h1 class='title'>Breast Cancer Awareness and Detection</h1>", unsafe_allow_html=True)

    # Create a container for the image and message
    with st.container():
        # Create two columns for layout
        col1, col2 = st.columns(2)

        # Load the image
        image_path = "1stpage.jpeg"  # Ensure the path is correct
        img = Image.open(image_path)

        # Display the image in the first column, resized
        with col1:
            st.image(img, caption='', use_column_width=False, width=400)  # Removed caption

        # Display the message in the second column
        with col2:
            st.markdown(
                """
                <div class='message'>
                Dear Woman,<br><br>
                You are strong, you are brave,<br>
                Your spirit unbroken, your will unchanged.<br>
                Cancer may test, but you'll prevail.<br>
                Every treatment a victory, every day a triumph.<br>
                You will rise, warrior, you will prevail.
                </div>
                """,
                unsafe_allow_html=True
            )
            


def second_page():
    # Custom CSS for styles
    st.markdown(
        """
        <style>
        .reportview-container, .main, .stApp {
            background-color: #FADADD !important;  /* Soothing pink background */
            height: 100vh !important; /* Makes the background cover the entire height */
        }
        .title {
            text-align: center;
            font-size: 48px;
            color: #003366;
            font-family: 'Arial', sans-serif;
            margin: 20px;
        }
        .message {
            font-size: 22px;  /* Reduced font size for better readability */
            color: #FFFFFF;  /* White color for better contrast */
            font-family: 'Georgia', serif;
            line-height: 1.5;  /* Adjusted line height for readability */
            background-color: rgba(0, 0, 0, 0.5);  /* Semi-transparent background */
            border-radius: 10px;  /* Rounded corners for the background */
            padding: 20px;  /* Padding for the message container */
            margin-top: 20px;  /* Space between image and message */
            max-height: 300px;  /* Maximum height for the message box */
            overflow-y: auto;  /* Allow scrolling if content exceeds the height */
            text-align: left;  /* Align the text to the left */
        }
        .container {
            display: flex;  /* Use flexbox to arrange items */
            justify-content: center;  /* Center horizontally */
            align-items: flex-start;  /* Align items to the start vertically */
            margin-top: 50px;
        }
        .image-container {
            flex: 1;  /* Allow image to take up equal space */
            display: flex;
            justify-content: center;  /* Center image */
            align-items: center;  /* Center vertically */
            margin-right: 20px;  /* Space between image and message */
        }
        .message-container {
            flex: 1;  /* Allow message to take up equal space */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title
    st.markdown("<h1 class='title'>Breast Cancer Awareness</h1>", unsafe_allow_html=True)

    # Create a container for the image and message
    with st.container():
        # Create two columns for layout
        col1, col2 = st.columns(2)

        # Load the image
        image_path = "2ndpage.jpg"  # Ensure the path is correct
        img = Image.open(image_path)

        # Display the image in the first column, maintaining aspect ratio
        with col1:
            st.image(img, caption='', use_column_width=True)  # Set use_column_width to True for responsiveness

        # Display the message in the second column
        with col2:
            st.markdown(
                """
                <div class='message'>
                Breast cancer is one of the most common cancers among women worldwide. Early detection is crucial in ensuring effective treatment and improving survival rates. Regular screenings, awareness of breast health, and understanding risk factors can empower women to take charge of their health. Remember, you are not alone in this fight. Together, we can spread awareness and support each other.Breast cancer remains one of the most prevalent cancers affecting women across the globe. However, with early detection, the chances of successful treatment and long-term survival increase significantly. Regular screenings such as mammograms, self-examinations, and a keen awareness of changes in breast health are essential in catching the disease in its early stages. Understanding personal risk factors, including family history and lifestyle, empowers women to make informed decisions about their health.Breast cancer is not just a physical battle but also an emotional journey, and no woman should feel like she is facing it alone. With the unwavering support of loved ones, healthcare professionals, and the global community, we can continue to raise awareness and foster an environment of compassion and hope. Together, we can break the silence, encourage early diagnosis, and remind every woman that there is strength in unity. Let’s continue to spread awareness, advocate for regular screenings, and offer support to those fighting breast cancer. Together, we can make a difference and inspire hope for a brighter, healthier future.
                </div>
                """,
                unsafe_allow_html=True
            )


def third_page():
    # Custom CSS for styles
    st.markdown(
        """
        <style>
        .reportview-container, .main, .stApp {
            background-color: #FADADD !important;  /* Soothing pink background */
            height: 100vh !important; /* Makes the background cover the entire height */
        }
        .title {
            text-align: center;
            font-size: 48px;
            color: #003366; /* Dark blue for the title */
            font-family: 'Arial', sans-serif;
            margin: 20px;
        }
        .sub-title {
            font-size: 30px;  /* Adjusted font size for subtitles */
            color: black;  /* Set subtitle color to black */
            font-family: 'Georgia', serif;
            margin: 10px 0; /* Space above and below the subtitle */
        }
        .content {
            font-size: 24px;  /* Font size for the content */
            color: #333333;  /* Dark gray for better contrast */
            font-family: 'Georgia', serif;
            line-height: 1.5;  /* Line height for readability */
            background-color: #B0E0E6;  /* Light yellow background */
            border-radius: 10px;  /* Rounded corners for the background */
            padding: 20px;  /* Padding for the content container */
            margin: 10px 0;  /* Margin around the content for spacing */
        }
        .more-info {
            color: #FF4500; /* Orange Red color for More Info */
            font-weight: bold; /* Make the text bold */
            cursor: pointer; /* Change cursor to pointer for interactivity */
            display: none; /* Hide the text in the interface */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title
    st.markdown("<h1 class='title'>Breast Cancer Detection</h1>", unsafe_allow_html=True)

    # Introduction to the Project
    st.markdown("<span class='sub-title'>Introduction to Our Project</span>", unsafe_allow_html=True)
    with st.expander("More Info", expanded=False):  # Dropdown menu with visible text
        st.markdown(
            "<div class='content'>"
            "Breast cancer is one of the leading causes of cancer-related deaths among women worldwide. Early detection significantly enhances treatment success and survival rates. Our project aims to leverage advanced imaging techniques to facilitate the early detection of breast cancer."
            "</div>",
            unsafe_allow_html=True
        )

   # Dataset Overview
    st.markdown("<span class='sub-title'>Dataset Overview</span>", unsafe_allow_html=True)
    with st.expander("More Info", expanded=False): 
       st.markdown(
           """
           <div class='content'>
           The dataset utilized in this project consists of 780 ultrasound images classified into three categories: normal, benign, and malignant. Each image has a corresponding ground truth mask image, providing pixel-level annotations necessary for effective training of our model.<br><br>
           - Subject Area: Medicine and Dentistry<br>
           - Specific Subject Area: Radiology and Imaging<br>
           - Type of Data: Images and Mask Images (PNG format)<br>
           - Data Acquisition: LOGIQ E9 and LOGIQ E9 Agile ultrasound systems<br>
           - Data Source Location: Baheya Hospital for Early Detection & Treatment of Women's Cancer, Cairo, Egypt<br>
           - Average Image Size: 500 × 500 pixels<br>
           </div>
           """,
           unsafe_allow_html=True
       )

   
    # Project Methodology
    st.markdown("<span class='sub-title'>Project Methodology</span>", unsafe_allow_html=True)
    with st.expander("More Info", expanded=False): 
        st.markdown(
            """
            <div class='content'>
            We implemented the Attention U-Net model, an advanced architecture known for its effectiveness in biomedical image segmentation. The Attention U-Net combines the strengths of U-Net with attention mechanisms, allowing the model to focus on relevant features while suppressing irrelevant information, which enhances segmentation accuracy.
            </div>
            """,
            unsafe_allow_html=True
        )
       # Team Members
    st.markdown("<span class='sub-title'>Project Team</span>", unsafe_allow_html=True)
    with st.expander("More Info", expanded=False): 
        st.markdown(
               """
               <div class='content'>
               <strong>Team Members:</strong> Pratiksha Sarvankar, Mahek Shaikh, Shahnawaz Shaikh.<br>
               This project was developed using Spyder IDE for coding and Streamlit for creating an interactive web application that showcases the results of our model and facilitates user engagement.
               </div>
               """,
               unsafe_allow_html=True
           )

def fourth_page():
    model_path = 'attention_unet_classification_model.keras'
    model = load_model(model_path, compile=False)

    # Define class labels
    class_labels = ["Benign", "Malignant", "Normal"]
    # Define a function for preprocessing the image
    def preprocess_image(image, target_size=(128, 128)):
        image = image.resize(target_size)
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    # Function to make predictions
    def make_prediction(image):
        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        predicted_index = np.argmax(prediction)

        # Ensure the predicted_index is within the bounds of class_labels
        if predicted_index < len(class_labels):
            predicted_class = class_labels[predicted_index]
            confidence_score = prediction[0][predicted_index] * 100
        else:
            predicted_class = "Unknown"
            confidence_score = 0.0

        return predicted_class, confidence_score
    def plot_confusion_matrix():
        fig, ax = plt.subplots()
        # Dummy data for demonstration; replace with your actual confusion matrix
        matrix = np.array([[80, 5, 5], [10, 70, 20], [5, 5, 90]])
        ax.matshow(matrix, cmap='Blues')
        for (i, j), val in np.ndenumerate(matrix):
            ax.text(j, i, f'{val}', ha='center', va='center')
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticklabels([''] + class_labels)
        ax.set_yticklabels([''] + class_labels)
        return fig
    def plot_roc_curve():
        fig, ax = plt.subplots()
        # Dummy data for demonstration; replace with your actual ROC curve data
        fpr = np.array([0, 0.1, 0.2, 1])
        tpr = np.array([0, 0.5, 0.75, 1])
        ax.plot(fpr, tpr, marker='o')
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        return fig
    def plot_precision_recall_curve():
        fig, ax = plt.subplots()
        # Dummy data for demonstration; replace with your actual Precision-Recall curve data
        precision = np.array([0.9, 0.85, 0.8, 0.95])
        recall = np.array([0.5, 0.7, 0.8, 0.9])
        ax.plot(recall, precision, marker='o')
        ax.set_title("Precision-Recall Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        return fig
    def plot_accuracy_over_epochs():
        fig, ax = plt.subplots()
        # Dummy data for demonstration; replace with your actual accuracy data
        epochs = np.arange(1, 11)
        accuracy = np.random.rand(10)  # Replace with your actual accuracy values
        ax.plot(epochs, accuracy, marker='o')
        ax.set_title("Accuracy Over Epochs")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        return fig
    def plot_loss_over_epochs():
        fig, ax = plt.subplots()
        # Dummy data for demonstration; replace with your actual loss data
        epochs = np.arange(1, 11)
        loss = np.random.rand(10)  # Replace with your actual loss values
        ax.plot(epochs, loss, marker='o')
        ax.set_title("Loss Over Epochs")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        return fig
    st.markdown(
        """
        <style>
            /* Set the main background color */
            .stApp {
                background-color: #FADADD;  /* Blush Pink */
            }
            /* Set the sidebar background color */
            .stSidebar {
                background-color: #B0E0E6;  /* Powder Blue */
            }
            h1 {
                color: #301934; 
                font-family: Arial; 
                font-size: 50px;
            }
            /* Set all output text to black */
            .black-text {
                color: black;
            }
        </style>
        """, unsafe_allow_html=True
    )
    st.markdown(
        "<h1 style='color:  #003366; font-family: Arial; font-size: 50px;'>Breast Cancer Detection</h1>",
        unsafe_allow_html=True,
    )
   # Sidebar for navigation
    st.sidebar.title("Navigation")

# Display a styled label for the file uploader
    st.sidebar.markdown(
        "<span style='color: black; font-size: 16px;'>Upload an image...</span>",
        unsafe_allow_html=True
    )

# File uploader without a label, as the styled label is displayed separately
    uploaded_file = st.sidebar.file_uploader("", type=["jpg", "jpeg", "png"])

# Display a header in the main section with black text
    st.markdown("<h2 style='color: black;'>Upload an image</h2>", unsafe_allow_html=True)


    # Dropdown for graph and prediction options
    # Change the color of the selectbox label using Markdown
    st.sidebar.markdown(
    "<span style='color: black; font-size: 16px;'>Choose analysis or prediction:</span>",
    unsafe_allow_html=True,
    ) 

    # Dropdown for graph and prediction options
    option = st.sidebar.selectbox(
    "",
    ["Prediction", "Confusion Matrix", 
     "ROC Curve", "Precision-Recall Curve", 
     "Accuracy Over Epochs", "Loss Over Epochs", "All Graphs"]
    )

    if uploaded_file is not None:
        # Open and resize the uploaded image
        image = Image.open(uploaded_file)
        resized_image = image.resize((200, 200))  # Resize to an ideal display size
        st.image(resized_image, caption='Uploaded Image', use_column_width=True)
        

        # Display the prediction if the "Prediction" option is selected
        if option == "Prediction":
            st.markdown("<h2 class='black-text'>Prediction</h2>", unsafe_allow_html=True)
            predicted_class, confidence_score = make_prediction(image)
            st.markdown(f"<span class='black-text'>Prediction: {predicted_class}</span>", unsafe_allow_html=True)
            st.markdown(f"<span class='black-text'>Confidence Score: {confidence_score:.2f}%</span>", unsafe_allow_html=True)

        # Display the selected analysis graph
        if option == "Confusion Matrix":
            st.markdown("<h2 class='black-text'>Confusion Matrix</h2>", unsafe_allow_html=True)
            st.pyplot(plot_confusion_matrix())

        elif option == "ROC Curve":
            st.markdown("<h2 class='black-text'>ROC Curve</h2>", unsafe_allow_html=True)
            st.pyplot(plot_roc_curve())

        elif option == "Precision-Recall Curve":
            st.markdown("<h2 class='black-text'>Precision-Recall Curve</h2>", unsafe_allow_html=True)
            st.pyplot(plot_precision_recall_curve())
            
        elif option == "Accuracy Over Epochs":
            st.markdown("<h2 class='black-text'>Accuracy Over Epochs</h2>", unsafe_allow_html=True)
            st.pyplot(plot_accuracy_over_epochs())

        elif option == "Loss Over Epochs":
            st.markdown("<h2 class='black-text'>Loss Over Epochs</h2>", unsafe_allow_html=True)
            st.pyplot(plot_loss_over_epochs())

        elif option == "All Graphs":
            st.markdown("<h2 class='black-text'>All Graphs</h2>", unsafe_allow_html=True)
            st.pyplot(plot_confusion_matrix())
            st.pyplot(plot_roc_curve())
            st.pyplot(plot_precision_recall_curve())
            st.pyplot(plot_accuracy_over_epochs())
            st.pyplot(plot_loss_over_epochs())

    else:
        st.markdown("<span class='black-text'>Please upload an image to proceed with the analysis.</span>", unsafe_allow_html=True)

    

    

# List of all page functions
pages = [first_page, second_page, third_page, fourth_page]

# Initialize session state for current page if not already set
if 'page' not in st.session_state:
    st.session_state.page = 0

# Display the current page based on session state
pages[st.session_state.page]()

# Navigation buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("◀️ Back"):
        if st.session_state.page > 0:
            st.session_state.page -= 1

with col2:
    if st.button("Next ▶️"):
        if st.session_state.page < len(pages) - 1:
            st.session_state.page += 1
