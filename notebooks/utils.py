#Requirements 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import seaborn as sns
import pycytominer
from copairs.map import average_precision
from copairs.map import mean_average_precision


#A function to run copairs, save csv and plot the mAP graph
def copairs_batches_newerversion(input_dict, negdiffby_parameters =[], filename = ''):
    """
    Runs copairs on the input profiles, saves the output CSVs and generates the overall mAP plot.
    input_dict: Takes dictionary as input
    negdiffby_parameters: takes list as an input. 
    filename: Takes string as input. It names the output csv based on the input
    """
    #defining the parameters for performing copairs
    pert_col = "Metadata_broad_sample"
    #control_col = "Metadata_control_type"
    #sister_compounds = 'Metadata_MoA'

    pos_sameby = [pert_col]
    pos_diffby = []

    neg_sameby = [] #control_col should be neg_sameby -this excludes the DMSO profiles for phenotypic distinctiveness

    neg_diffby = negdiffby_parameters

    #neg_diffby varies based on whether the mAP needs to be calculated with respect to the controls or treatments 
    #if mAP == 'Control':
        #neg_diffby=[control_col]
    #else:
        #neg_diffby=[pert_col]
    
    batch_size = 20000
    null_size = 10000

    output_dict = {}
    for i in input_dict:
         
        with open(i, 'rb') as filetype:
            if filetype.read(2) == b'\x1f\x8b':
                df = pd.read_csv(i, compression='gzip')
            else:
                df = pd.read_csv(i)
        df['Metadata_control_type'].fillna('trt', inplace=True)
        df['Metadata_broad_sample'].fillna('control', inplace=True)
        if 'Metadata_control_type' not in negdiffby_parameters:
            df = df.query("Metadata_control_type == 'trt'")
        else:
            df = df
        name = input_dict.get(i)
        metadata_columns = [ c for c in df.columns if 'Metadata' in c]
        feature_columns = [c for c in df.columns if not 'Metadata' in c]
      
        meta = df[metadata_columns].copy()

        features = df[feature_columns]
        features = features.dropna(axis=1).values
        result = average_precision(meta, features, pos_sameby, pos_diffby, neg_sameby, neg_diffby, batch_size)
        result.to_csv(f"{i[:-4]}_Result_NegconNorm_mAP_{filename}.csv")

        aggregated_mAP = mean_average_precision(result, sameby=pos_sameby, null_size= 10000, threshold=0.05, seed=2)
        output_dict[name] = aggregated_mAP

        output_dict[name].to_csv(f"{i[:-4]}_Aggregate_result_NegconNorm_mAP_{filename}.csv")


    #making a violin plot from the data
    combined_df = pd.DataFrame()
    for i in output_dict.keys():
        df = output_dict.get(i)
        combined_df = pd.concat([combined_df, df.assign(dataset = i)])

    df = combined_df    
    fig = px.violin(x=df['mean_average_precision'], y=df['dataset'], range_x=[0,1.02], box=True, color_discrete_sequence=['#0173b2','#de8f05','#029e73','#d55e00','#cc78bc'], color=df['dataset'], orientation='h')
    fig.update_traces(points='all', width=0.5)
    fig.update_layout(height=2500, width=2750, font_family='sans serif', font=dict(size=95, color='black'), xaxis_title='Mean Average Precision', yaxis_title='', showlegend=False)
    fig.update_layout({'plot_bgcolor':'rgba(0,0,0,0)'})
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', tickangle=90)
    fig.data = fig.data[::-1] 

    fig.show()

    return fig, output_dict

#A function to split Broad ID
def BRD_ID(i):
    if type(i) != float:
        ID = i.split('-')
        if len(ID) >1:
            return ID[1]
        
#A function to calculate the cell count from the metadata and divide it by 100 
# There was a difference in the column name in the plate 49 and it was renamed
def cell_count_norm_colorscheme(dict1, dict2):
    #reading the moa metadata file 
    moa_metadata = pd.read_csv('..\\copairs_csv\\LC00009948_MoA_Common_Names.csv')
    new_dict = {}
    for i in dict1:  
        raw_df = pd.read_csv(i)
        if 'phasefeatures' in i:
            raw_df = raw_df.rename(columns={'Metadata_broad_sample.1':'Metadata_BRD ID'})
            test_df = raw_df.groupby('Metadata_BRD ID')['Metadata_Count_Cells'].mean().to_frame()
            test_df = test_df.reset_index()
            test_df = test_df.rename(columns={'Metadata_BRD ID':'BRD ID','Metadata_Count_Cells':'Metadata_Count_Cells'+dict1[i]})
            test_df['Metadata_Count_Cells'+dict1[i]+'_norm'] = test_df['Metadata_Count_Cells'+dict1[i]]/100
            new_dict.update({dict1.get(i):test_df})
        else:
            test_df = raw_df.groupby('Metadata_BRD ID')['Metadata_Count_Cells'].mean().to_frame()
            test_df = test_df.reset_index()
            test_df = test_df.rename(columns={'Metadata_BRD ID':'BRD ID','Metadata_Count_Cells':'Metadata_Count_Cells'+dict1[i]})
            test_df['Metadata_Count_Cells'+dict1[i]+'_norm'] = test_df['Metadata_Count_Cells'+dict1[i]]/100
            new_dict.update({dict1.get(i):test_df})

    
    cell_count_df = None
    for key, df in new_dict.items():
        if cell_count_df is None:
            cell_count_df = df.copy()
        else:
            cell_count_df = pd.merge(cell_count_df, df, on='BRD ID')


    output_dict = {}
    for i in dict2:
        raw_df = pd.read_csv(i)
        raw_df = raw_df.rename(columns={'mean_average_precision':'mean_average_precision'+dict2.get(i)})
        subset_df = raw_df[['Metadata_broad_sample', 'mean_average_precision'+dict2.get(i)]]
        output_dict[dict2.get(i)] = subset_df


    combined_df = None 
    for key, df in output_dict.items():
        if combined_df is None:
            combined_df = df.copy()
        else:
            combined_df = pd.merge(combined_df, df, on='Metadata_broad_sample')
            
    
    combined_df['BRD ID'] = combined_df['Metadata_broad_sample'].map(BRD_ID)
    

    combined_df_metadata = pd.merge(combined_df, moa_metadata, on='BRD ID')
    combined_df_metadata_cell_count = pd.merge(combined_df_metadata, cell_count_df, on='BRD ID')
    combined_df_metadata_cell_count = combined_df_metadata_cell_count.rename(columns={'mean_average_precisionStandard CP':'Standard CP', 'mean_average_precisionCP + MitoBrilliant':'CP + MitoBrilliant', 'mean_average_precisionCP + Phenovue phalloidin 400LS':'CP + Phenovue phalloidin 400LS', 'mean_average_precisionStandard CP (exposed to ChromaLive)':'Standard CP (exposed to ChromaLive)', 'mean_average_precisionChromaLive + Hoechst':'ChromaLive + Hoechst'})

    plot = go.Figure()
    
    plot.add_trace(go.Scatter(x=combined_df_metadata_cell_count['MoA'], y=combined_df_metadata_cell_count['Standard CP'],hovertext=combined_df_metadata_cell_count['Common Name'], mode='markers',marker=dict(color='#0173b2'), name = 'Standard CP', marker_size=combined_df_metadata_cell_count['Metadata_Count_CellsStandard CP_norm']))
    plot.add_trace(go.Scatter(x=combined_df_metadata_cell_count['MoA'], y=combined_df_metadata_cell_count['CP + MitoBrilliant'],hovertext=combined_df_metadata_cell_count['Common Name'], mode='markers',marker=dict(color='#de8f05'), name = 'CP + MitoBrilliant', marker_size=combined_df_metadata_cell_count['Metadata_Count_CellsCP + MitoBrilliant_norm']))
    plot.add_trace(go.Scatter(x=combined_df_metadata_cell_count['MoA'], y=combined_df_metadata_cell_count['CP + Phenovue phalloidin 400LS'],hovertext=combined_df_metadata_cell_count['Common Name'], mode='markers',marker=dict(color='#029e73'), name = 'CP + Phenovue phalloidin 400LS', marker_size=combined_df_metadata_cell_count['Metadata_Count_CellsCP + Phenovue phalloidin 400LS_norm']))
    plot.add_trace(go.Scatter(x=combined_df_metadata_cell_count['MoA'], y=combined_df_metadata_cell_count['Standard CP (exposed to ChromaLive)'],hovertext=combined_df_metadata_cell_count['Common Name'], mode='markers',marker=dict(color='#d55e00'), name = 'Standard CP (exposed to ChromaLive)', marker_size=combined_df_metadata_cell_count['Metadata_Count_CellsStandard CP (exposed to ChromaLive)_norm']))
    plot.add_trace(go.Scatter(x=combined_df_metadata_cell_count['MoA'], y=combined_df_metadata_cell_count['ChromaLive + Hoechst'],hovertext=combined_df_metadata_cell_count['Common Name'], mode='markers',marker=dict(color='#cc78bc'), name = 'ChromaLive + Hoechst', marker_size=combined_df_metadata_cell_count['Metadata_Count_CellsChromaLive + Hoechst_norm']))

    plot.update_layout(height=1250,width=2000, font_family='sans serif', font=dict(size=24, color='Black'), boxmode='group',yaxis_title = 'Mean average precision',  legend=dict(yanchor="bottom",y=0,xanchor="right",x=1))
    plot.update_layout({'plot_bgcolor':'rgba(0,0,0,0)'})
    plot.update_xaxes(tickangle=90, categoryorder='total ascending')
    plot.update_traces(marker_sizemin=3, marker_sizemode='area', marker_sizeref=0.1)
    plot.update_xaxes(linecolor='black')
    plot.update_yaxes(linecolor='black')
    #plot.show('notebook')

    return plot, combined_df_metadata_cell_count

# A function to remove phase features and run the feature selection
def phase_removal_and_feature_selection (file=''):

    df = pd.read_csv(file, compression='gzip')
    print(df.shape)
    features_wo_phase = [c for c in df.columns if not "Phase" in c]
    df_wo_phase = df[features_wo_phase]
    print(df_wo_phase.shape)

    feature_selected_df = pycytominer.feature_select(profiles=df_wo_phase, operation=['variance_threshold', 'correlation_threshold', 'drop_na_columns', 'blocklist'])

    return feature_selected_df

#A function to run copairs, save csv and plot the mAP graph
def copairs_batches_earlytimepoint(input_dict, mAP=''):
    
    #defining the parameters for performing copairs
    pert_col = "Metadata_broad_sample"
    control_col = "Metadata_control_type"

    pos_sameby = [pert_col]
    pos_diffby = []

    neg_sameby = []

    #neg_diffby varies based on whether the mAP needs to be calculated with respect to the controls or treatments 
    if mAP == 'Control':
        neg_diffby=[control_col]
    else:
        neg_diffby=[pert_col]
    
    batch_size = 20000
    null_size = 10000

    output_dict = {}
    for i in input_dict:
         
        with open(i, 'rb') as filetype:
            if filetype.read(2) == b'\x1f\x8b':
                df = pd.read_csv(i, compression='gzip')
            else:
                df = pd.read_csv(i)
        name = input_dict.get(i)
        metadata_columns = [ c for c in df.columns if 'Metadata' in c]
        feature_columns = [c for c in df.columns if not 'Metadata' in c]

        meta = df[metadata_columns].copy()

        meta['Metadata_control_type'] = meta['Metadata_control_type'].fillna('trt')
        meta['Metadata_broad_sample'] = meta['Metadata_broad_sample'].fillna('control')
        features = df[feature_columns]
        features = features.dropna(axis=1).values
        result = average_precision(meta, features, pos_sameby, pos_diffby, neg_sameby, neg_diffby, batch_size)
        result.to_csv(f"{i[:-4]}_Result_NegconNorm_mAP_wrt_{mAP}.csv")

        aggregated_mAP = mean_average_precision(result, sameby=pos_sameby, null_size= 10000, threshold=0.05, seed=2)
        output_dict[name] = aggregated_mAP

        output_dict[name].to_csv(f"{i[:-4]}_Aggregate_result_NegconNorm_mAP_wrt_{mAP}.csv")


    #making a violin plot from the data
    combined_df = pd.DataFrame()
    for i in output_dict.keys():
        df = output_dict.get(i)
        combined_df = pd.concat([combined_df, df.assign(dataset = i)])

    df = combined_df    
    fig = px.violin(x=df['mean_average_precision'], y=df['dataset'], range_x=[0,1.02], box=True, color_discrete_sequence=['#0173b2','#de8f05','#029e73','#d55e00','#cc78bc'], color=df['dataset'], orientation='h')
    fig.update_traces(points='all', width=0.5)
    fig.update_layout(height=2500, width=2750, font_family='sans serif', font=dict(size=95, color='black'), xaxis_title='Mean Average Precision', yaxis_title='', showlegend=False)
    fig.update_layout({'plot_bgcolor':'rgba(0,0,0,0)'})
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', tickangle=90)
    fig.data = fig.data[::-1] 

    fig.show()

    return fig, output_dict

def cell_count_norm_colorscheme_early(dict1, dict2):
    #reading the moa metadata file 
    moa_metadata = pd.read_csv('..\\copairs_csv\\LC00009948_MoA_Common_Names.csv')
    new_dict = {}
    for i in dict1:  
        raw_df = pd.read_csv(i)
        if 'phasefeatures' in i:
            raw_df = raw_df.rename(columns={'Metadata_broad_sample.1':'Metadata_BRD ID'})
            test_df = raw_df.groupby('Metadata_BRD ID')['Metadata_Count_Cells'].mean().to_frame()
            test_df = test_df.reset_index()
            test_df = test_df.rename(columns={'Metadata_BRD ID':'BRD ID','Metadata_Count_Cells':'Metadata_Count_Cells'+dict1[i]})
            test_df['Metadata_Count_Cells'+dict1[i]+'_norm'] = test_df['Metadata_Count_Cells'+dict1[i]]/100
            new_dict.update({dict1.get(i):test_df})
        else:
            test_df = raw_df.groupby('Metadata_BRD ID')['Metadata_Count_Cells'].mean().to_frame()
            test_df = test_df.reset_index()
            test_df = test_df.rename(columns={'Metadata_BRD ID':'BRD ID','Metadata_Count_Cells':'Metadata_Count_Cells'+dict1[i]})
            test_df['Metadata_Count_Cells'+dict1[i]+'_norm'] = test_df['Metadata_Count_Cells'+dict1[i]]/100
            new_dict.update({dict1.get(i):test_df})

    
    cell_count_df = None
    for key, df in new_dict.items():
        if cell_count_df is None:
            cell_count_df = df.copy()
        else:
            cell_count_df = pd.merge(cell_count_df, df, on='BRD ID')


    output_dict = {}
    for i in dict2:
        raw_df = pd.read_csv(i)
        raw_df = raw_df.rename(columns={'mean_average_precision':'mean_average_precision'+dict2.get(i)})
        subset_df = raw_df[['Metadata_broad_sample', 'mean_average_precision'+dict2.get(i)]]
        output_dict[dict2.get(i)] = subset_df


    combined_df = None 
    for key, df in output_dict.items():
        if combined_df is None:
            combined_df = df.copy()
        else:
            combined_df = pd.merge(combined_df, df, on='Metadata_broad_sample')
            
    
    combined_df['BRD ID'] = combined_df['Metadata_broad_sample'].map(BRD_ID)
    

    combined_df_metadata = pd.merge(combined_df, moa_metadata, on='BRD ID')
    combined_df_metadata_cell_count = pd.merge(combined_df_metadata, cell_count_df, on='BRD ID')
    combined_df_metadata_cell_count = combined_df_metadata_cell_count.rename(columns={'mean_average_precisionChromaLive+DRAQ7+Cas3/7_4h':'ChromaLive+DRAQ7+Cas3/7_4h', 'mean_average_precisionChromaLive+DRAQ7+Cas3/7_24h':'ChromaLive+DRAQ7+Cas3/7_24h', 'mean_average_precisionChromaLive_4h':'ChromaLive_4h', 'mean_average_precisionChromaLive_24h':'ChromaLive_24h', 'mean_average_precisionChromaLive_48h':'ChromaLive_48h'})

    plot = go.Figure()
    
    plot.add_trace(go.Scatter(x=combined_df_metadata_cell_count['MoA'], y=combined_df_metadata_cell_count['ChromaLive+DRAQ7+Cas3/7_4h'],hovertext=combined_df_metadata_cell_count['Common Name'], mode='markers',marker=dict(color=px.colors.qualitative.Set2[2], symbol='triangle-up'), name = 'ChromaLive+DRAQ7+Cas3/7_4h',marker_opacity =0.5,  marker_size=combined_df_metadata_cell_count['Metadata_Count_CellsChromaLive+DRAQ7+Cas3/7_4h_norm']))
    plot.add_trace(go.Scatter(x=combined_df_metadata_cell_count['MoA'], y=combined_df_metadata_cell_count['ChromaLive+DRAQ7+Cas3/7_24h'],hovertext=combined_df_metadata_cell_count['Common Name'], mode='markers',marker=dict(color=px.colors.qualitative.Set2[2], symbol='square'), name = 'ChromaLive+DRAQ7+Cas3/7_24h',marker_opacity =0.5, marker_size=combined_df_metadata_cell_count['Metadata_Count_CellsChromaLive+DRAQ7+Cas3/7_24h_norm']))
    plot.add_trace(go.Scatter(x=combined_df_metadata_cell_count['MoA'], y=combined_df_metadata_cell_count['ChromaLive_4h'],hovertext=combined_df_metadata_cell_count['Common Name'], mode='markers',marker=dict(color=px.colors.qualitative.Set2[3], symbol='triangle-up'), name = 'ChromaLive_4h',marker_opacity =0.5, marker_size=combined_df_metadata_cell_count['Metadata_Count_CellsChromaLive_4h_norm']))
    plot.add_trace(go.Scatter(x=combined_df_metadata_cell_count['MoA'], y=combined_df_metadata_cell_count['ChromaLive_24h'],hovertext=combined_df_metadata_cell_count['Common Name'], mode='markers',marker=dict(color=px.colors.qualitative.Set2[3], symbol='square'), name = 'ChromaLive_24h',marker_opacity =0.5, marker_size=combined_df_metadata_cell_count['Metadata_Count_CellsChromaLive_24h_norm']))
    plot.add_trace(go.Scatter(x=combined_df_metadata_cell_count['MoA'], y=combined_df_metadata_cell_count['ChromaLive_48h'],hovertext=combined_df_metadata_cell_count['Common Name'], mode='markers',marker=dict(color=px.colors.qualitative.Set2[3], symbol='pentagon'), name = 'ChromaLive_48h',marker_opacity =0.5, marker_size=combined_df_metadata_cell_count['Metadata_Count_CellsChromaLive_48h_norm']))

    

    plot.update_layout(height=1250,width=2000, font_family='sans serif', font=dict(size=24, color='Black'), boxmode='group',yaxis_title = 'Mean average precision',  legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01))
    plot.update_layout({'plot_bgcolor':'rgba(0,0,0,0)'})
    plot.update_xaxes(tickangle=90, categoryorder='total ascending')
    plot.update_traces(marker_sizemin=10, marker_sizemode='area',marker_sizeref=1)
    plot.update_layout(legend=dict(itemsizing="constant"))
    plot.update_xaxes(linecolor='black')
    plot.update_yaxes(linecolor='black')
    #plot.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)')
    plot.show('notebook')

    return plot, combined_df_metadata_cell_count

#only for subsection of the plots
def cell_count_norm_colorscheme_early_part2(dict1, dict2):
    #reading the moa metadata file 
    moa_metadata = pd.read_csv('..\\copairs_csv\\LC00009948_MoA_Common_Names.csv')
    new_dict = {}
    for i in dict1:  
        raw_df = pd.read_csv(i)
        if 'phasefeatures' in i:
            raw_df = raw_df.rename(columns={'Metadata_broad_sample.1':'Metadata_BRD ID'})
            test_df = raw_df.groupby('Metadata_BRD ID')['Metadata_Count_Cells'].mean().to_frame()
            test_df = test_df.reset_index()
            test_df = test_df.rename(columns={'Metadata_BRD ID':'BRD ID','Metadata_Count_Cells':'Metadata_Count_Cells'+dict1[i]})
            test_df['Metadata_Count_Cells'+dict1[i]+'_norm'] = test_df['Metadata_Count_Cells'+dict1[i]]/100
            new_dict.update({dict1.get(i):test_df})
        else:
            test_df = raw_df.groupby('Metadata_BRD ID')['Metadata_Count_Cells'].mean().to_frame()
            test_df = test_df.reset_index()
            test_df = test_df.rename(columns={'Metadata_BRD ID':'BRD ID','Metadata_Count_Cells':'Metadata_Count_Cells'+dict1[i]})
            test_df['Metadata_Count_Cells'+dict1[i]+'_norm'] = test_df['Metadata_Count_Cells'+dict1[i]]/100
            new_dict.update({dict1.get(i):test_df})

    
    cell_count_df = None
    for key, df in new_dict.items():
        if cell_count_df is None:
            cell_count_df = df.copy()
        else:
            cell_count_df = pd.merge(cell_count_df, df, on='BRD ID')


    output_dict = {}
    for i in dict2:
        raw_df = pd.read_csv(i)
        raw_df = raw_df.rename(columns={'mean_average_precision':'mean_average_precision'+dict2.get(i)})
        subset_df = raw_df[['Metadata_broad_sample', 'mean_average_precision'+dict2.get(i)]]
        output_dict[dict2.get(i)] = subset_df


    combined_df = None 
    for key, df in output_dict.items():
        if combined_df is None:
            combined_df = df.copy()
        else:
            combined_df = pd.merge(combined_df, df, on='Metadata_broad_sample')
            
    
    combined_df['BRD ID'] = combined_df['Metadata_broad_sample'].map(BRD_ID)
    

    combined_df_metadata = pd.merge(combined_df, moa_metadata, on='BRD ID')
    combined_df_metadata_cell_count = pd.merge(combined_df_metadata, cell_count_df, on='BRD ID')
    combined_df_metadata_cell_count = combined_df_metadata_cell_count.rename(columns={'mean_average_precisionChromaLive_4h':'ChromaLive_4h', 'mean_average_precisionChromaLive_24h':'ChromaLive_24h', 'mean_average_precisionChromaLive_48h':'ChromaLive_48h'}) #'mean_average_precisionChromaLive_4h':'ChromaLive_4h', 'mean_average_precisionChromaLive_24h':'ChromaLive_24h', 'mean_average_precisionChromaLive_48h':'ChromaLive_48h'})

    #to add different colors to the compounds belonging to the same MoA
    combined_df_metadata_cell_count['Compound_Number'] = combined_df_metadata_cell_count.groupby('MoA').cumcount() + 1

    # possible colorblind friendly colors - ['#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc', '#ca9161', '#fbafe4', '#949494', '#ece133', '#56b4e9']
    color_map = {1:'#56b4e9', 2:'#fbafe4'}
    plot = go.Figure()
    
    #plot.add_trace(go.Scatter(x=combined_df_metadata_cell_count['MoA'], y=combined_df_metadata_cell_count['ChromaLive+DRAQ7_4h'],hovertext=combined_df_metadata_cell_count['Common Name'], mode='markers',marker=dict(color=combined_df_metadata_cell_count['Compound_Number'].map(color_map), symbol='triangle-up', size = 20), name = 'ChromaLive+DRAQ7_4h',marker_opacity =0.5, legendgroup='group'))  #marker_size=combined_df_metadata_cell_count['Metadata_Count_CellsChromaLive+DRAQ7_4h_norm']))
    #plot.add_trace(go.Scatter(x=combined_df_metadata_cell_count['MoA'], y=combined_df_metadata_cell_count['ChromaLive+DRAQ7_24h'],hovertext=combined_df_metadata_cell_count['Common Name'], mode='markers',marker=dict(color=combined_df_metadata_cell_count['Compound_Number'].map(color_map), symbol='square', size = 20), name = 'ChromaLive+DRAQ7_24h',marker_opacity =0.5, legendgroup='group2')) #marker_size=combined_df_metadata_cell_count['Metadata_Count_CellsChromaLive+DRAQ7_24h_norm']))
    plot.add_trace(go.Scatter(x=combined_df_metadata_cell_count['MoA'], y=combined_df_metadata_cell_count['ChromaLive_4h'],hovertext=combined_df_metadata_cell_count['Common Name'], mode='markers',marker=dict(color=combined_df_metadata_cell_count['Compound_Number'].map(color_map), symbol='triangle-up', size = 20), name = 'ChromaLive_4h',marker_opacity =0.5, showlegend=False)) #marker_size=combined_df_metadata_cell_count['Metadata_Count_CellsChromaLive_4h_norm']))
    plot.add_trace(go.Scatter(x=combined_df_metadata_cell_count['MoA'], y=combined_df_metadata_cell_count['ChromaLive_24h'],hovertext=combined_df_metadata_cell_count['Common Name'], mode='markers',marker=dict(color=combined_df_metadata_cell_count['Compound_Number'].map(color_map), symbol='square', size = 20), name = 'ChromaLive_24h',marker_opacity =0.5, showlegend=False)) #marker_size=combined_df_metadata_cell_count['Metadata_Count_CellsChromaLive_24h_norm']))
    plot.add_trace(go.Scatter(x=combined_df_metadata_cell_count['MoA'], y=combined_df_metadata_cell_count['ChromaLive_48h'],hovertext=combined_df_metadata_cell_count['Common Name'], mode='markers',marker=dict(color=combined_df_metadata_cell_count['Compound_Number'].map(color_map), symbol='pentagon', size = 20), name = 'ChromaLive_48h',marker_opacity =0.5, showlegend=False)) #marker_size=combined_df_metadata_cell_count['Metadata_Count_CellsChromaLive_48h_norm']))

    #Adding custom legends 
    plot.add_trace(go.Scatter(x=[None], y=[None], mode='markers',marker=dict(color='#56b4e9', symbol='triangle-up', size=12),legendgroup='group1',showlegend=True,name='Compound 1 - ChromaLive_4h'))
    plot.add_trace(go.Scatter(x=[None], y=[None], mode='markers',marker=dict(color='#56b4e9', symbol='square', size=12),legendgroup='group2',showlegend=True,name='Compound 1 - ChromaLive_24h'))
    plot.add_trace(go.Scatter(x=[None], y=[None], mode='markers',marker=dict(color='#56b4e9', symbol='pentagon', size=12),legendgroup='group3',showlegend=True,name='Compound 1 - ChromaLive_48h'))

    plot.add_trace(go.Scatter(x=[None], y=[None],mode='markers',marker=dict(color='#fbafe4', symbol='triangle-up', size=12),legendgroup='group4',showlegend=True,name='Compound 2 - ChromaLive_4h'))
    plot.add_trace(go.Scatter(x=[None], y=[None],mode='markers',marker=dict(color='#fbafe4', symbol='square', size=12),legendgroup='group5',showlegend=True,name='Compound 2 - ChromaLive_24h'))
    plot.add_trace(go.Scatter(x=[None], y=[None],mode='markers',marker=dict(color='#fbafe4', symbol='pentagon', size=12),legendgroup='group6',showlegend=True,name='Compound 2 - ChromaLive_48h'))


    plot.update_layout(height=1250,width=2000, font_family='sans serif', font=dict(size=26, color='Black'), boxmode='group',yaxis_title = 'Mean average precision',  legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01))
    plot.update_layout({'plot_bgcolor':'rgba(0,0,0,0)'})
    plot.update_xaxes(tickangle=90, categoryorder='total ascending')
    #plot.update_traces(marker_sizemin=10, marker_sizemode='area',marker_sizeref=1)
    plot.update_layout(legend=dict(itemsizing="constant"))
    plot.update_xaxes(linecolor='black')
    plot.update_yaxes(linecolor='black')
    #plot.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)')
    plot.show('notebook')

    return plot, combined_df_metadata_cell_count

#To compare the mAP across dye sets 
def scatter_plot_comparison_colorblack(df, col1='', col2='', cutoff=''):

    df['difference'] = abs(df[col1]-df[col2]) 
    val = df['difference'] 
    df['Modified MoA'] = df['difference'].apply(lambda x: 'Other' if x < cutoff else df.loc[df['difference']==x, 'MoA'].iloc[0])
   

    plot = go.Figure()
    plot = px.scatter(x=df[col1],y= df[col2],labels = {'x':'Mean average precision - '+ col1, 'y':'Mean average precision - ' + col2}, color=df['Modified MoA'], color_discrete_map={True:px.colors.qualitative.Set2, 'Other':'darkgrey'}, hover_name=df["MoA"])
    plot.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(color='lightgrey', dash='dashdot'))
    plot.update_layout(legend=dict(orientation="h"), height=1200, width=1000,font_family='sans serif', font=dict(size=26, color='Black'), plot_bgcolor='rgba(0,0,0,0)',)
    plot.update_xaxes(ticks='outside', showline=True, linecolor='black')
    plot.update_yaxes(ticks='outside',showline=True, linecolor='black')
    plot.update_traces(marker=dict(size=14))
    #plot.show('notebook')

    return plot

#Plot with trendline
def scatter_plot_comparison_colorblack_trendline(df, col1='', col2=''):

    df['difference'] = abs(df[col1]-df[col2]) 
    val = df['difference'] 
    df['Modified MoA'] = df['difference'].apply(lambda x: 'Other' if x < 0.3 else df.loc[df['difference']==x, 'MoA'].iloc[0])
   

    plot = go.Figure()
    plot = px.scatter(x=df[col1],y= df[col2],labels = {'x':'Mean average precision - '+ col1, 'y':'Mean average precision - ' + col2}, color=df['Modified MoA'], color_discrete_map={True:px.colors.qualitative.Set2, 'Other':'lightgrey'}, hover_name=df["MoA"], trendline='ols', trendline_scope="overall")
    plot.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(color='grey', dash='dashdot'))
    plot.update_layout(legend=dict(orientation="h"), height=800, width=1000,font_family='sans serif', font=dict(size=20, color='Black'))
    #plot.show('notebook')

    return plot

def plot_precision_values_dyeset(df, list=[], rename_columns = {}):
    output_dict = {}
    grouped_df = df.groupby('MoA')
    for i in list:
        df_MoA = grouped_df.get_group(i)
        if df_MoA.shape == (2,20):
            col1 = df_MoA.iloc[0,8]
            col2 = df_MoA.iloc[1,8]
            df_MoA = df_MoA.rename(columns=rename_columns)
            df_MoA_subset = df_MoA.iloc[:,1:6].transpose()
            df_MoA_subset = df_MoA_subset.reset_index()
            df_MoA_subset = df_MoA_subset.rename(columns={'index':'Dye sets', df_MoA_subset.columns.values[1]:col1, df_MoA_subset.columns.values[2]:col2})
            output_dict[i] = df_MoA_subset

        else:
            col1 = df_MoA.iloc[0,8]
            df_MoA = df_MoA.rename(columns=rename_columns)
            df_MoA_subset = df_MoA.iloc[:,1:6].transpose()
            df_MoA_subset = df_MoA_subset.reset_index()
            df_MoA_subset = df_MoA_subset.rename(columns={'index':'Dye sets', df_MoA_subset.columns.values[1]:col1})
            output_dict[i] = df_MoA_subset
    
        
    figure = make_subplots(rows=10, cols=5,y_title='Mean Average Precision', subplot_titles = [i for i in output_dict.keys()], shared_xaxes=True, horizontal_spacing=0.02, vertical_spacing=0.02)
        
    
    MoA_list = [key for key in output_dict]
    x=0
    #to_be_removed = [90, 91, 92, 93, 94, 95]
    
    for i in range(1,11):
        for j in range(1,6):
            MoA = MoA_list[x]
            df = output_dict.get(MoA)
            
            if df.shape == (5,3):
                if [i,j] not in [[10,3], [10, 4], [10,5]]:
                    figure.add_trace(go.Scatter(x=df['Dye sets'], y= df.iloc[:, 1], mode='markers',marker=dict(color='#0173b2'), name=df.columns.values[1],showlegend=False), row=i, col=j)
                    figure.add_trace(go.Scatter(x=df['Dye sets'], y=df.iloc[:, 2],mode= 'markers',marker=dict(color='#de8f05'), name=df.columns.values[2],showlegend=False),row=i, col=j)
                    figure.add_annotation(xref='x domain', yref='y domain', x=0.1, y=0.95, text = df.columns.values[1],font=dict(color='#0173b2'), showarrow=False, row=i, col=j)
                    figure.add_annotation(xref='x domain', yref='y domain', x=0.9, y=0.95, text = df.columns.values[2],font=dict(color='#de8f05'), showarrow=False, row=i, col=j)
                
            else:
                if [i,j] not in [[10,3], [10, 4], [10,5]]:
                    figure.add_trace(go.Scatter(x=df['Dye sets'], y= df.iloc[:, 1], mode='markers', marker=dict(color='#0173b2'), name=df.columns.values[1],showlegend=False), row=i, col=j)
                    figure.add_annotation(xref='x domain', yref='y domain', x=0.5, y=0.95, text = df.columns.values[1],font=dict(color='#0173b2'), showarrow=False, row=i, col=j)


            if x < (len(MoA_list)-1):
                x= x+1 
                    
            else:
                pass
            
    figure.update_layout(height= 2500, width =2000, font_family='sans serif',font=dict(size=19, color='Black'), yaxis_title_font=dict(size=34))
    figure.update_yaxes(tickfont=dict(size=24), range=[0,1.3])
    figure.update_traces(marker=dict(size=12))
    figure.update_xaxes(tickfont=dict(size=24), tickangle = 90)
    figure.update_annotations(font_size=19)
    figure.update_layout({'plot_bgcolor':'rgba(0,0,0,0)'})
    figure.update_xaxes(linecolor='black')
    figure.update_yaxes(linecolor='black')

    

    return figure, output_dict

#To calculate fraction retrieved
def different_threshold_metric (input_dict={}, threshold_value = []):
    threshold_dict = {}
    threshold_df_combined = pd.DataFrame()
    for values in threshold_value:
        for i in input_dict:
            name = input_dict.get(i)
            aggregate_df = pd.read_csv(i)
            aggregate_df['compound_passed_threshold'] = aggregate_df['corrected_p_value'].apply(lambda x: True if x < values else False)
            threshold_dict[name] = (aggregate_df['compound_passed_threshold'].sum()/90) * 100
            threshold_df = pd.DataFrame.from_dict(threshold_dict, orient='index')
            threshold_df = threshold_df.rename(columns={0:values})
        
        threshold_df_combined = pd.concat([threshold_df_combined, threshold_df], axis =1)
    threshold_df_combined = threshold_df_combined.transpose()

    return threshold_df_combined





