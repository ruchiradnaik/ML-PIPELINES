import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
import json

# Import functions from pipeline.py
from pipeline import preprocess_data, train_and_evaluate_models

st.title("📂 AutoML Pipeline - Upload, Preprocess & Train Models")

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

session_dir = f"session_{st.session_state.session_id}"
os.makedirs(session_dir, exist_ok=True)

# --------------------------
# File Upload
# --------------------------
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Save file in session dir
        file_path = os.path.join(session_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load dataframe
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        # Save dataframe as pickle
        df_pickle_path = os.path.join(session_dir, "dataframe.pkl")
        df.to_pickle(df_pickle_path)

        st.success("✅ File uploaded successfully!")
        st.write(f"📁 Session ID: {st.session_state.session_id}")
        st.write("### Preview of Dataset:")
        st.dataframe(df.head())

        # --------------------------
        # Configuration
        # --------------------------
        st.write("### 🔧 Configure Your ML Problem")
        all_columns = df.columns.tolist()
        X_cols = st.multiselect("Select Independent Features (X)", options=all_columns)
        y_col = st.selectbox("Select Target Column (y)", options=all_columns)
        task_type = st.radio("Choose Task Type", ["Classification", "Regression"])

        if X_cols and y_col and y_col not in X_cols:
            # Save session config
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

            with open("latest_session.txt", "w") as f:
                f.write(st.session_state.session_id)

            st.success(f"Configuration saved for session: {st.session_state.session_id}")

            # --------------------------
            # Run Preprocessing
            # --------------------------
            if st.button("Run Preprocessing"):
                with st.spinner("Preprocessing your data..."):
                    try:
                        X_train, X_test, y_train, y_test, report, encoders, scaler = preprocess_data(
                            df, X_cols, y_col, task_type.lower()
                        )
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        st.session_state.encoders = encoders
                        st.session_state.scaler = scaler
                        st.success("✅ Preprocessing complete!")
                    except Exception as e:
                        st.error(f"❌ Preprocessing failed: {str(e)}")

                # Show report as a table
                if 'X_train' in st.session_state:
                    st.write("### 🔎 Preprocessing Report")
                    report_df = pd.DataFrame(list(report.items()), columns=["Metric", "Value"])
                    st.table(report_df)

            # --------------------------
            # Train Models
            # --------------------------
            if 'X_train' in st.session_state:
                if st.button("Train Models"):
                    with st.spinner("Training multiple models... This may take a moment."):
                        try:
                            models, results = train_and_evaluate_models(
                                st.session_state.X_train, st.session_state.y_train,
                                st.session_state.X_test, st.session_state.y_test,
                                task_type.lower()
                            )
                            st.session_state.models = models
                            st.session_state.results = results
                            st.success("✅ Model training complete!")
                        except Exception as e:
                            st.error(f"❌ Training failed: {str(e)}")

                # Show results
                if 'results' in st.session_state:
                    st.write("### 📊 Model Evaluation Results")
                    results_df = pd.DataFrame(st.session_state.results).T
                    if task_type.lower() == "classification":
                        results_df = results_df.sort_values(by='Accuracy', ascending=False)
                    else:
                        results_df = results_df.sort_values(by='R2', ascending=False)
                    st.table(results_df)

                    # --------------------------
                    # Export Selected Model
                    # --------------------------
                    selected_model = st.selectbox("Select Model to Export", list(st.session_state.results.keys()))
                    if st.button("Export Model"):
                        with st.spinner("Preparing export..."):
                            try:
                                bundle = {
                                    'model': st.session_state.models[selected_model],
                                    'scaler': st.session_state.scaler,
                                    'encoders': st.session_state.encoders,
                                    'task_type': task_type.lower(),
                                    'feature_columns': X_cols,
                                    'target_column': y_col
                                }

                                # Save with joblib
                                model_path = os.path.join(session_dir, f"{selected_model}.joblib")
                                joblib.dump(bundle, model_path)

                                # Download
                                with open(model_path, 'rb') as f:
                                    st.download_button(
                                        label="📥 Download Model",
                                        data=f.read(),
                                        file_name=f"{selected_model}.joblib",
                                        mime="application/octet-stream"
                                    )

                                st.success(f"✅ Ready to download: {selected_model}.joblib")
                                st.info("""
                                **How to Use the Model:**
                                ```python
                                import joblib
                                import pandas as pd

                                # Load bundle
                                bundle = joblib.load("your_model.joblib")

                                # Preprocess new data
                                new_df = pd.DataFrame(...)  # your new data with same columns
                                X_new = new_df[bundle['feature_columns']]

                                # Apply encoders
                                for col, le in bundle['encoders'].items():
                                    if col in X_new.columns:
                                        X_new[col] = le.transform(X_new[col].astype(str))

                                # Scale
                                scaler = bundle['scaler']
                                X_new_scaled = scaler.transform(X_new)

                                # Predict
                                model = bundle['model']
                                predictions = model.predict(X_new_scaled)

                                # Decode if classification
                                if bundle['task_type'] == 'classification' and bundle['target_column'] in bundle['encoders']:
                                    predictions = bundle['encoders'][bundle['target_column']].inverse_transform(predictions)

                                print(predictions)
                                ```
                                """)
                            except Exception as e:
                                st.error(f"❌ Export failed: {str(e)}")

        else:
            if y_col in X_cols:
                st.warning("⚠️ Target column cannot be in features. Please adjust selections.")

    except Exception as e:
        st.error(f"❌ File upload failed: {str(e)}")
