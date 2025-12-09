import pandas as pd 
import numpy as np 
import plotly.express as px 
import plotly.graph_objects as go
from scipy import stats

def filter_proteins_by_presence(df, min_presence_pct=0.7):
    total_samples = df.shape[1]
    min_samples = int(np.ceil(min_presence_pct * total_samples))
    presence_counts = df.notna().sum(axis=1)
    filter_mask = presence_counts >= min_samples
    filtered_df = df[filter_mask]
    return filtered_df

def prepare_ID_datasets (data, nreps = 3, loc_col = 'PTM_localization', loc_treshold = 0.75):
    num_full = []
    num_class_I_all = []
    num_rest_all = []
    for dataset in data:
        num_class_I = []
        num_rest = []
        id_col = []
        tmp_df1 = dataset.iloc[:,:nreps]
        tmp_df2 = dataset[dataset[loc_col]>=loc_treshold].iloc[:,:nreps]
        for col in tmp_df1.columns:
            num_full.append(len(tmp_df1[col].dropna()))
        for col in tmp_df2.columns:
            num_full.append(len(tmp_df2[col].dropna()))
        for col in tmp_df1:
            num_class_I.append(len(tmp_df2[col].dropna()))
            num_rest.append(len(tmp_df1[col].dropna()) - len(tmp_df2[col].dropna()))
        num_class_I_all.append(num_class_I)
        num_rest_all.append(num_rest)
    for i in range(1, len(data)+1):
        id_col.append('dataset'+str(i))
    id_col1 = np.repeat(id_col, nreps*2).tolist()
    df_all = pd.DataFrame({'Number': num_full, 'ID':id_col1})
    return df_all, num_class_I_all, num_rest_all, id_col

def plot_ID_barplot(strip_numbers, classI_numbers, rest_numbers, ids, classI_color, 
                    rest_color, point_color, point_size, 
                    point_jitter, plot_width, plot_height, plot_template, bar_width):

    fig = px.strip(strip_numbers, y = 'Number', x = 'ID')
    fig.update_traces(jitter = point_jitter,
                      marker = dict(size = point_size, color = point_color, line = dict(width = 0.5, color = 'black')))
    for i, el in enumerate(classI_numbers):
        fig.add_trace(go.Bar(y = [np.mean(el)], x = [ids[i]], width = bar_width, marker_line_color = 'black', marker_color = classI_color))
        fig.add_trace(go.Bar(y = [np.mean(rest_numbers[i])], x = [ids[i]], width = bar_width, marker_line_color = 'black', marker_color = rest_color))
    fig.update_yaxes(range = [0, max(strip_numbers['Number']) + 0.05*max(strip_numbers['Number'])])
    fig.update_layout(width = plot_width, height = plot_height, template = plot_template, barmode = 'stack', showlegend = False)
    return fig

def get_cumulative_barplot (list_of_datasets, nreps, loc_col = 'PTM_localization', loc_treshold = 0.75, classI_color = '#8A0000', 
                    rest_color = '#E83F25', point_color = '#393E46', point_size = 9, 
                    point_jitter = 1, plot_width = 600, plot_height = 600, plot_template = 'plotly_white', bar_width = 0.5, get_cumulative_dataset = False):
    a,b,c,d = prepare_ID_datasets(data=list_of_datasets, nreps=nreps, loc_col=loc_col, loc_treshold=loc_treshold)
    e = plot_ID_barplot(strip_numbers=a, classI_numbers=b, rest_numbers=c, ids = d, 
                        classI_color=classI_color, rest_color=rest_color, point_color=point_color, point_size=point_size,
                        point_jitter=point_jitter, plot_width=plot_width, plot_height=plot_height, plot_template=plot_template, bar_width=bar_width)
    if get_cumulative_dataset == False:
        return e
    else:
        return e,a

def calculate_interdilution_correlation(dataset_list, minimal_dilution_dataset, xaxis_values, nreps = 3, log2 = False):
    lookup_dicts = []
    for i, el in enumerate(dataset_list):
        if minimal_dilution_dataset.equals(el) == True:
            dataset_list.pop(i)
    
    for df in dataset_list:
        lookup_dict = {}
        for i, row in df.iterrows():
            key=row['PTM_Collapse_key']
            if log2 != False:
                mean_value=np.mean(list(2,row[:nreps]))
            mean_value=np.mean(list(np.power(2,row[:nreps])))
            lookup_dict[key] = mean_value
        lookup_dicts.append(lookup_dict)
    tmp = []
    keys = minimal_dilution_dataset['PTM_Collapse_key'].values
    if log2 != False:
        ref_values = np.mean(2,minimal_dilution_dataset.iloc[:,:3].values, axis = 1)
    ref_values = np.mean(np.power(2,minimal_dilution_dataset.iloc[:,:3]).values, axis = 1)
    
        

    for i in range(len(minimal_dilution_dataset)):
        row_data = []
        ptm_key = keys[i]
        ref_value = ref_values[i]
        
        for lookup_dict in lookup_dicts:
            if ptm_key in lookup_dict:
                row_data.append(lookup_dict[ptm_key])
            else:
                found = False
                for dict_key in lookup_dict:
                    if ptm_key in dict_key:
                        row_data.append(lookup_dict[dict_key])
                        found = True
                        break
                    if not found:
                        row_data.append(np.nan)
        row_data.append(ref_value)
        tmp.append(row_data)
    
    r_squared_values = []
    mean_intensity = []

    for inner_list in tmp:
        valid_pairs = [(xaxis_values[i], val) for i, val in enumerate(inner_list) if i < len(xaxis_values) and not pd.isna(val)]
        
        if len(valid_pairs) >= np.round(len(dataset_list)+1, 0):
            x = [pair[0] for pair in valid_pairs]
            y = [pair[1] for pair in valid_pairs]
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
            
            r_squared = r_value ** 2
            r_squared_values.append(r_squared)
            mean_intensity.append(np.mean(inner_list))
    return r_squared_values