# %% [markdown]
# # 📊 Smart Data Loader & Analyst (Excel & PDF) with Google Gemini AI
#
# This Streamlit app intelligently loads data from files and allows you to
# ask questions about the content in a conversational chat interface.

# %%
import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import json
import openpyxl
from typing import Dict, Tuple, Optional, List
import re
from io import BytesIO, StringIO
import warnings
import fitz  # PyMuPDF for PDF processing

warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1. Page Configuration and API Key Setup

st.set_page_config(
    page_title="Smart Data Analyst",
    page_icon="🤖",
    layout="wide"
)

st.title("📊 Smart Data Loader & Analyst")
st.markdown("Upload an Excel or PDF file to automatically extract data, then ask questions about it.")

# --- API Keys Hardcoded as requested ---
# Key for file processing and data extraction
FILE_PROCESSING_API_KEY = "AIzaSyCYF6F8fB7fSEJoyNPg4zvXhiO2RpcK8M8"
# Separate key for the conversational chat analyst
CHAT_API_KEY = "AIzaSyBC7o13qrbPWy0r4UwjBDX3z0g2GoIdmw0"


# %% [markdown]
# ## 2. Smart Excel Loader Class
# (The core logic for analyzing and loading Excel files)

class SmartExcelLoader:
    """
    Hybrid Excel loader using Google Gemini AI.
    """
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        if 'structure_cache' not in st.session_state:
            st.session_state.structure_cache = {}

    def analyze_sheet_structure(self, file_data: bytes, sheet_name: str) -> Dict:
        # This method uses AI to find the starting row and structure of an Excel sheet
        try:
            df_sample = pd.read_excel(BytesIO(file_data), sheet_name=sheet_name, nrows=30, header=None)
        except Exception as e:
            st.error(f"Error reading sheet {sheet_name}: {e}")
            return self._get_default_structure()
        sample_str = df_sample.head(20).to_string()
        cache_key = f"{st.session_state.get('filename', 'unknown')}_{sheet_name}_{df_sample.shape}"
        if cache_key in st.session_state.structure_cache:
            return st.session_state.structure_cache[cache_key]
        prompt = f"""
        Analyze this Excel data. Return ONLY a JSON object:
        {{"data_start_row": <number>, "sheet_type": "tabular" or "report", "has_merged_cells": true or false}}
        Sheet: {sheet_name}\nData:\n{sample_str}
        """
        try:
            with st.spinner(f"🤖 AI is analyzing '{sheet_name}'..."):
                response = self.model.generate_content(prompt)
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                structure = json.loads(json_match.group()) if json_match else {}
                structure['header_rows'] = [structure.get('data_start_row', 1) - 1]
                st.session_state.structure_cache[cache_key] = structure
                return structure
        except Exception:
            return self._get_fallback_structure(df_sample, sheet_name)

    def _get_fallback_structure(self, df_sample: pd.DataFrame, sheet_name: str) -> Dict:
        # Fallback logic for Excel sheets if AI fails
        data_start_row = 1
        for idx, row in df_sample.iterrows():
            if row.apply(lambda x: isinstance(x, (int, float)) and pd.notna(x)).sum() >= 3:
                data_start_row = idx + 1
                break
        return {'data_start_row': data_start_row, 'has_merged_cells': False}

    def _get_default_structure(self) -> Dict:
        return {'data_start_row': 1, 'has_merged_cells': False}

    def load_sheet(self, file_data: bytes, sheet_name: str, use_ai: bool = True) -> Optional[pd.DataFrame]:
        # Loads a single Excel sheet
        structure = self.analyze_sheet_structure(file_data, sheet_name) if use_ai else self._get_default_structure()
        try:
            df = pd.read_excel(BytesIO(file_data), sheet_name=sheet_name, skiprows=structure.get('data_start_row', 1) - 1, engine='openpyxl')
            return self._apply_cleaning(df, structure)
        except Exception as e:
            st.error(f"❌ Error loading {sheet_name}: {e}")
            return None

    def _apply_cleaning(self, df: pd.DataFrame, structure: Dict) -> pd.DataFrame:
        df = df.dropna(how='all').dropna(axis=1, how='all').reset_index(drop=True)
        if structure.get('has_merged_cells', False):
            df.iloc[:, 0] = df.iloc[:, 0].fillna(method='ffill')
        df.columns = [str(col).strip() for col in df.columns]
        # Final fix for all Arrow errors
        return df.fillna('').astype(str)

# %% [markdown]
# ## 3. PDF & Chat Analyst Functions

def process_pdf(api_key: str, file_data: bytes) -> Optional[Dict[str, pd.DataFrame]]:
    """ Extracts all tabular data from a PDF using Gemini AI. """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    try:
        doc = fitz.open(stream=file_data, filetype="pdf")
        full_text = "".join(page.get_text() for page in doc)
        doc.close()
    except Exception as e:
        st.error(f"Failed to read PDF file: {e}")
        return None

    prompt = f"""
    You are an expert financial data analyst. Your task is to extract ALL tables from the following text.
    Return a single JSON object where keys are table titles and values are the table data as a CSV string.
    If no tables are found, return an empty JSON object {{}}.

    PDF Text to Analyze:
    ---
    {full_text[:15000]}
    ---
    """
    try:
        with st.spinner("🤖 AI is scanning the PDF for all tables..."):
            response = model.generate_content(prompt)
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if not json_match: return None
            json_data = json.loads(json_match.group())
            dataframes = {}
            for title, csv_string in json_data.items():
                df = pd.read_csv(StringIO(csv_string))
                # Final fix for all Arrow errors
                dataframes[title] = df.fillna('').astype(str)
            return dataframes
    except Exception as e:
        st.error(f"PDF data extraction failed: {e}")
        return None

def generate_data_context(dataframes: Dict[str, pd.DataFrame]) -> str:
    """ Creates a string representation of all loaded data for the chat AI. """
    context = ""
    for name, df in dataframes.items():
        context += f"--- DataFrame: {name} ---\n"
        context += df.to_string(max_rows=10)
        context += "\n\n"
    return context

# %% [markdown]
# ## 4. Main Streamlit Application UI

# Initialize session state for chat and data
if 'messages' not in st.session_state: st.session_state.messages = []
if 'loaded_dfs' not in st.session_state: st.session_state.loaded_dfs = None
if 'data_context' not in st.session_state: st.session_state.data_context = None
if 'current_file_id' not in st.session_state: st.session_state.current_file_id = None

uploaded_file = st.file_uploader("📁 Select your Excel or PDF file:", type=['xlsx', 'xls', 'pdf'])

if uploaded_file:
    # Check if a new file has been uploaded to reset state
    if st.session_state.current_file_id != uploaded_file.file_id:
        st.session_state.messages = []
        st.session_state.loaded_dfs = None
        st.session_state.data_context = None
        st.session_state.current_file_id = uploaded_file.file_id

    st.sidebar.success(f"File Uploaded: \n**{uploaded_file.name}**")
    file_data = uploaded_file.getvalue()

    # --- File Processing Section ---
    if uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
        st.header("Excel File Processor", divider='rainbow')
        loader = SmartExcelLoader(api_key=FILE_PROCESSING_API_KEY)
        xl_file = pd.ExcelFile(BytesIO(file_data))
        sheet_name = st.selectbox("Select a sheet to analyze:", xl_file.sheet_names, key=f"sb_{uploaded_file.file_id}")
        if st.button(f"Load & Analyze Sheet: '{sheet_name}'", type="primary"):
            df = loader.load_sheet(file_data, sheet_name)
            if df is not None:
                st.success("Data loaded successfully!")
                st.dataframe(df)
                st.session_state.loaded_dfs = {sheet_name: df}

    elif uploaded_file.name.lower().endswith('.pdf'):
        st.header("PDF File Processor", divider='rainbow')
        if st.button("📄 Scan and Extract All Tables from PDF", type="primary"):
            extracted_tables = process_pdf(FILE_PROCESSING_API_KEY, file_data)
            if extracted_tables:
                st.success(f"✅ Successfully extracted {len(extracted_tables)} tables!")
                st.session_state.loaded_dfs = extracted_tables
                for title, df in extracted_tables.items():
                    with st.expander(f"**Table: {title}**", expanded=False):
                        st.dataframe(df)

    # --- Context Generation (runs once after data is loaded) ---
    if st.session_state.loaded_dfs and st.session_state.data_context is None:
        st.session_state.data_context = generate_data_context(st.session_state.loaded_dfs)

# --- Conversational Analyst Chat UI Section ---
if st.session_state.data_context:
    st.header("💬 Conversational Data Analyst", divider='rainbow')

    genai.configure(api_key=CHAT_API_KEY)
    # --- FIX: Changed "gemini-pro" to a current, supported model ---
    chat_model = genai.GenerativeModel("gemini-1.5-flash")
    # -----------------------------------------------------------

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare the prompt for the AI
        full_prompt = f"You are a helpful data analyst. Answer the user's question based ONLY on the following data context.\n\n[Data Context]\n{st.session_state.data_context}\n\n[Question]\n{prompt}"

        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = chat_model.generate_content(full_prompt)
                response_text = response.text
                st.markdown(response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})

else:
    if uploaded_file is None:
        st.info("👋 Welcome! Please upload a file to begin.")