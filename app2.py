import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import functions from pipeline.py
from pipeline import preprocess_data, train_and_evaluate_models

# Page config
st.set_page_config(
    page_title="AutoML Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 AutoML Pipeline</h1>
    <p>Upload your data, configure your ML problem, and train models with ease!</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

session_dir = f"session_{st.session_state.session_id}"
os.makedirs(session_dir, exist_ok=True)

# Sidebar for session info and progress
with st.sidebar:
    st.header("📋 Session Info")
    st.info(f"**Session ID:** {st.session_state.session_id}")
    
    # Progress tracker
    st.header("📈 Progress")
    progress_steps = ["Upload Data", "Configure", "Preprocess", "Train Models", "Export"]
    current_step = 0
    
    if 'df' in st.session_state:
        current_step = 1
    if 'config_set' in st.session_state:
        current_step = 2
    if 'preprocessing_done' in st.session_state:
        current_step = 3
    if 'models_trained' in st.session_state:
        current_step = 4
    if 'model_exported' in st.session_state:
        current_step = 5
    
    for i, step in enumerate(progress_steps):
        if i < current_step:
            st.success(f"✅ {step}")
        elif i == current_step:
            st.info(f"🔄 {step}")
        else:
            st.text(f"⏳ {step}")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Step 1: File Upload
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("📁 Step 1: Upload Your Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx"],
        help="Upload your dataset to get started. Supported formats: CSV, Excel"
    )

    if uploaded_file is not None:
        try:
            # Save file in session dir
            file_path = os.path.join(session_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load dataframe
            with st.spinner("Loading your dataset..."):
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)

            # Save dataframe
            df_pickle_path = os.path.join(session_dir, "dataframe.pkl")
            df.to_pickle(df_pickle_path)
            st.session_state.df = df

            st.success("✅ Dataset uploaded successfully!")
            
            # Dataset overview
            st.subheader("📊 Dataset Overview")
            
            # Create metrics in columns
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("Rows", f"{len(df):,}")
            with metric_cols[1]:
                st.metric("Columns", len(df.columns))
            with metric_cols[2]:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            with metric_cols[3]:
                missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
                st.metric("Missing Data", f"{missing_pct:.1f}%")

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.stop()

    st.markdown('</div>', unsafe_allow_html=True)

    # Step 2: Configuration (only show if file is uploaded)
    if 'df' in st.session_state:
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.header("⚙️ Step 2: Configure Your ML Problem")
        
        df = st.session_state.df
        all_columns = df.columns.tolist()
        
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            task_type = st.radio(
                "Choose Task Type",
                ["Classification", "Regression"],
                help="Classification for predicting categories, Regression for predicting continuous values"
            )
            
            y_col = st.selectbox(
                "Select Target Column (what you want to predict)",
                options=[""] + all_columns,
                help="This is the column you want to predict"
            )
        
        with config_col2:
            available_features = [col for col in all_columns if col != y_col] if y_col else all_columns
            X_cols = st.multiselect(
                "Select Features (input variables)",
                options=available_features,
                default=available_features[:5] if len(available_features) >= 5 else available_features,
                help="These columns will be used to make predictions"
            )

        if X_cols and y_col and y_col not in X_cols:
            # Save configuration
            session_data = {
                'dataframe_path': df_pickle_path,
                'file_path': file_path,
                'X_cols': X_cols,
                'y_col': y_col,
                'task_type': task_type,
                'columns': all_columns
            }

            with open(os.path.join(session_dir, "session_data.json"), 'w') as f:
                json.dump(session_data, f, indent=2)

            st.session_state.config_set = True
            st.success("✅ Configuration saved!")

        elif y_col in X_cols:
            st.warning("⚠️ Target column cannot be in features. Please adjust your selection.")

        st.markdown('</div>', unsafe_allow_html=True)

        # Step 3: Preprocessing
        if st.session_state.get('config_set', False):
            st.markdown('<div class="step-container">', unsafe_allow_html=True)
            st.header("🔄 Step 3: Data Preprocessing")
            
            if st.button("🚀 Run Preprocessing", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("Starting preprocessing...")
                    progress_bar.progress(20)
                    
                    X_train, X_test, y_train, y_test, report, encoders, scaler = preprocess_data(
                        df, X_cols, y_col, task_type.lower()
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("Preprocessing complete!")
                    
                    # Store in session state
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.encoders = encoders
                    st.session_state.scaler = scaler
                    st.session_state.preprocessing_done = True
                    
                    st.success("✅ Preprocessing completed successfully!")
                    
                    # Display preprocessing report
                    st.subheader("📋 Preprocessing Report")
                    report_data = []
                    for key, value in report.items():
                        if isinstance(value, (int, float)):
                            report_data.append({"Metric": key, "Value": f"{value:,.0f}" if isinstance(value, int) else f"{value:.4f}"})
                        else:
                            report_data.append({"Metric": key, "Value": str(value)})
                    
                    report_df = pd.DataFrame(report_data)
                    st.table(report_df)
                    
                except Exception as e:
                    st.error(f"❌ Preprocessing failed: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()

            st.markdown('</div>', unsafe_allow_html=True)

        # Step 4: Model Training
        if st.session_state.get('preprocessing_done', False):
            st.markdown('<div class="step-container">', unsafe_allow_html=True)
            st.header("🎯 Step 4: Train ML Models")
            
            if st.button("🤖 Train Models", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("Training multiple models...")
                    progress_bar.progress(30)
                    
                    models, results = train_and_evaluate_models(
                        st.session_state.X_train, st.session_state.y_train,
                        st.session_state.X_test, st.session_state.y_test,
                        task_type.lower()
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("Model training complete!")
                    
                    st.session_state.models = models
                    st.session_state.results = results
                    st.session_state.models_trained = True
                    
                    st.success("✅ All models trained successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()

            st.markdown('</div>', unsafe_allow_html=True)

        # Step 5: Results and Export
        if st.session_state.get('models_trained', False):
            st.markdown('<div class="step-container">', unsafe_allow_html=True)
            st.header("📊 Step 5: Model Results & Export")
            
            results = st.session_state.results
            results_df = pd.DataFrame(results).T
            
            # Sort results by best metric
            if task_type.lower() == "classification":
                results_df = results_df.sort_values(by='Accuracy', ascending=False)
                best_metric = "Accuracy"
            else:
                results_df = results_df.sort_values(by='R2', ascending=False)
                best_metric = "R2"
            
            # Display best model prominently
            best_model = results_df.index[0]
            best_score = results_df.iloc[0][best_metric]
            
            st.success(f"🏆 Best Model: **{best_model}** with {best_metric}: **{best_score:.4f}**")
            
            # Show results table
            st.subheader("📈 All Model Results")
            st.dataframe(results_df.style.highlight_max(axis=0), use_container_width=True)
            
            # Model selection for export
            st.subheader("📦 Export Model")
            selected_model = st.selectbox(
                "Choose model to export:",
                list(results.keys()),
                index=0,
                help="Select the model you want to download and use for predictions"
            )
            
            col_export1, col_export2 = st.columns(2)
            
            with col_export1:
                if st.button("📥 Export Selected Model", type="primary", use_container_width=True):
                    try:
                        with st.spinner("Preparing model for export..."):
                            bundle = {
                                'model': st.session_state.models[selected_model],
                                'scaler': st.session_state.scaler,
                                'encoders': st.session_state.encoders,
                                'task_type': task_type.lower(),
                                'feature_columns': X_cols,
                                'target_column': y_col
                            }

                            model_path = os.path.join(session_dir, f"{selected_model}.joblib")
                            joblib.dump(bundle, model_path)
                            
                            with open(model_path, 'rb') as f:
                                st.download_button(
                                    label=f"📥 Download {selected_model}.joblib",
                                    data=f.read(),
                                    file_name=f"{selected_model}.joblib",
                                    mime="application/octet-stream",
                                    use_container_width=True
                                )
                            
                            st.session_state.model_exported = True
                            st.success("✅ Model ready for download!")
                            
                    except Exception as e:
                        st.error(f"❌ Export failed: {str(e)}")
            
            with col_export2:
                if st.button("📋 Show Usage Instructions", use_container_width=True):
                    st.session_state.show_instructions = True
            
            # Usage instructions
            if st.session_state.get('show_instructions', False):
                st.subheader("💡 How to Use Your Exported Model")
                st.code(f"""
import joblib
import pandas as pd

# Load the exported model bundle
bundle = joblib.load("{selected_model}.joblib")

# Prepare new data (same structure as training data)
new_data = pd.DataFrame({{
    # Add your new data here with columns: {X_cols}
}})

X_new = new_data[bundle['feature_columns']]

# Apply the same preprocessing
for col, encoder in bundle['encoders'].items():
    if col in X_new.columns:
        X_new[col] = encoder.transform(X_new[col].astype(str))

# Scale the features
X_new_scaled = bundle['scaler'].transform(X_new)

# Make predictions
predictions = bundle['model'].predict(X_new_scaled)

# For classification, decode predictions back to original labels
if bundle['task_type'] == 'classification' and bundle['target_column'] in bundle['encoders']:
    predictions = bundle['encoders'][bundle['target_column']].inverse_transform(predictions)

print("Predictions:", predictions)
                """, language="python")

            st.markdown('</div>', unsafe_allow_html=True)

# Right column - Data insights and preview
with col2:
    if 'df' in st.session_state:
        st.header("📋 Data Insights")
        
        df = st.session_state.df
        
        # Dataset preview
        with st.expander("🔍 Dataset Preview", expanded=True):
            st.dataframe(df.head(), use_container_width=True)
        
        # Data types
        with st.expander("🏷️ Column Types"):
            dtype_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Non-Null': df.notnull().sum(),
                'Unique': df.nunique()
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        # Missing values visualization
        if df.isnull().sum().sum() > 0:
            with st.expander("❓ Missing Values"):
                missing_data = df.isnull().sum()
                missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
                
                if len(missing_data) > 0:
                    fig = px.bar(
                        x=missing_data.values,
                        y=missing_data.index,
                        orientation='h',
                        title="Missing Values by Column"
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Target distribution (if configured)
        if st.session_state.get('config_set', False) and y_col:
            with st.expander("🎯 Target Distribution"):
                if df[y_col].dtype in ['object', 'category']:
                    # Categorical target
                    value_counts = df[y_col].value_counts()
                    fig = px.pie(values=value_counts.values, names=value_counts.index)
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Numerical target
                    fig = px.histogram(df, x=y_col, title=f"Distribution of {y_col}")
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Built with ❤️ using Streamlit | AutoML Pipeline v2.0</div>",
    unsafe_allow_html=True
)