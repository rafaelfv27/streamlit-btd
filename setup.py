import pickle
import streamlit as st
import numpy as np
from pandas import DataFrame, read_csv
from rdkit import Chem
from rdkit.Chem import PandasTools, AllChem, DataStructs, Draw
from streamlit_ketcher import st_ketcher


class MoleculePropertyPrediction:
    def __init__(self):
        self.dataset = self.read_dataset()  
        self.model = self.load_model()  
   
    def read_dataset(self):
        dataset = read_csv('models/no_missing_data.csv')
        print(dataset.head())
        return dataset

    def load_model(self):
        with open('models/model_xgb_512_3.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    
    def process_input(self, smiles, etn, max_abs):
        input_data = DataFrame({'smiles': [smiles], 'etn': [etn], 'max_abs': [max_abs]})
        
        PandasTools.AddMoleculeColumnToFrame(input_data, smilesCol='smiles')
        morgan = [AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=512) for m in input_data['ROMol']]
        morgan = DataFrame([list(fp) for fp in morgan])
        morgan.columns = ['MF_' + str(i) for i in range(morgan.shape[1])]
        
        morgan['etn'] = input_data['etn'].values
        morgan['max_abs'] = input_data['max_abs'].values
        
        return morgan
    
    def predict(self, morgan_fingerprint):
        prediction = self.model.predict(morgan_fingerprint)
        return prediction
    
    def calculate_similarity(self, user_smiles):
        similarity_list = []
        user_mol = AllChem.MolFromSmiles(user_smiles)
        user_fp = AllChem.GetMorganFingerprint(user_mol, 2)

        for i, row in self.dataset.iterrows():
            dataset_mol = AllChem.MolFromSmiles(row['smiles'])
            dataset_fp = AllChem.GetMorganFingerprint(dataset_mol, 2)
            similarity = DataStructs.TanimotoSimilarity(user_fp, dataset_fp)
            similarity_list.append(similarity) 

        self.dataset['Similarity'] = similarity_list
        most_similar_molecules = self.dataset.sort_values(by='Similarity', ascending=False).head(10)
        return most_similar_molecules

    def draw_most_similar_molecule(self, top_similar_molecules):
        st.write('Most Similar Molecule:')
        smiles = top_similar_molecules.iloc[0]['smiles']
        mol = Chem.MolFromSmiles(smiles)
        Draw.MolToImage(mol)
        st.image(Draw.MolToImage(mol), use_column_width=True)

    def main(self):
        st.title('Molecule Property Prediction with Similarity')
        
        drawn_smiles = st_ketcher()
        solvent = st.selectbox('Select Solvent:', sorted(self.dataset['solvent'].unique()), format_func=lambda x: f"{x} ({self.dataset.loc[self.dataset['solvent'] == x, 'etn'].values[0]})")
        etn = self.dataset.loc[self.dataset['solvent'] == solvent, 'etn'].values[0]
        max_abs = st.number_input('Enter the Max Abs value:', min_value=0.0, value=0.0, step=0.01)
        
        if st.button('Predict'):
            if drawn_smiles:
                morgan_input = self.process_input(drawn_smiles, etn, max_abs)
                prediction = self.predict(morgan_input)
                st.write(f'Predicted Value: {prediction}')
                top_similar_molecules = self.calculate_similarity(drawn_smiles) 
                st.write('The most similar molecule:')
                st.write(top_similar_molecules[['smiles', 'Similarity']] * 100)
                self.draw_most_similar_molecule(top_similar_molecules)
            else:
                st.error('Please draw a valid molecule.')

if __name__ == '__main__':
    app = MoleculePropertyPrediction()
    app.main()
