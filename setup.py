import pickle
import streamlit as st 
from pandas import DataFrame
from rdkit.Chem import PandasTools, AllChem
from streamlit_ketcher import st_ketcher



def load_model():
    with open('model/model_xgb_512_3.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


def process_input(smiles, etn, max_abs):
    input_data = DataFrame({'smiles': [smiles], 'etn': [etn], 'max_abs': [max_abs]})
    
    
    PandasTools.AddMoleculeColumnToFrame(input_data, smilesCol='smiles')
    
    
    morgan = [AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=512) for m in input_data['ROMol']]
    morgan = [list(i) for i in morgan]
    morgan = DataFrame(morgan)
    morgan.columns = ['MF_' + str(i) for i in morgan.columns]
    
    
    morgan['etn'] = input_data['etn'].values
    morgan['max_abs'] = input_data['max_abs'].values

    return morgan


def drawing_smiles():
    smiles = st_ketcher()
    etn = st.number_input('ETN Value', min_value=0.0, step=0.1)
    max_abs = st.number_input('Max Absorption (max_abs)', min_value=0.0, step=0.1)
    if st.button('Predict'):
        if smiles:
            prediction = predict(smiles, etn, max_abs)
            st.write(f'Prediction: {prediction}')
        else:
            st.error('Please enter a valid SMILES string.')


def predict(smiles, etn, max_abs):
    model = load_model()
    data = process_input(smiles, etn, max_abs)
    prediction = model.predict(data)
    return prediction


def main():
    st.title('Molecule Property Prediction')
    drawing_smiles()


if __name__ == '__main__':
    main()

    
