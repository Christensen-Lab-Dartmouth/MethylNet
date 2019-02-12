from umap import UMAP
import plotly.graph_objs as go
import plotly.offline as py
import pandas as pd
import numpy as np
import networkx as nx
import click
import pickle
from sklearn.preprocessing import LabelEncoder

CONTEXT_SETTINGS = dict(help_option_names=['-h','--help'], max_content_width=90)

@click.group(context_settings= CONTEXT_SETTINGS)
@click.version_option(version='0.1')
def visualize():
    pass

def umap_embed(beta_df, outcome_col, n_neighbors, supervised=False, min_dist=0.1, metric='euclidean'):
    umap=UMAP(n_components=3, random_state=42, metric=metric, n_neighbors=n_neighbors, min_dist=min_dist)
    t_data=pd.DataFrame(umap.fit_transform(beta_df) if not supervised else umap.fit_transform(beta_df,LabelEncoder().fit_transform(outcome_col)),index=beta_df.index,columns=['x','y','z'])
    print(outcome_col,t_data)
    t_data['color']=outcome_col
    return t_data

def plotly_plot(t_data_df, output_fname, G=None, axes_off=False):
    plots = []
    if t_data_df['color'].dtype == np.float64:
        plots.append(
            go.Scatter3d(x=t_data_df['x'], y=t_data_df['y'],
                         z=t_data_df['z'],
                         name='', mode='markers',
                         marker=dict(color=t_data_df['color'], size=2, colorscale='Viridis',
                         colorbar=dict(title='Colorbar')), text=t_data_df['color'] if 'name' not in list(t_data_df) else t_data_df['name']))
    else:
        colors = t_data_df['color'].unique()
        c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, len(colors) + 2)]
        color_dict = {name: c[i] for i,name in enumerate(sorted(colors))}

        for name,col in color_dict.items():
            print(name,col)
            plots.append(
                go.Scatter3d(x=t_data_df['x'][t_data_df['color']==name], y=t_data_df['y'][t_data_df['color']==name],
                             z=t_data_df['z'][t_data_df['color']==name],
                             name=str(name), mode='markers',
                             marker=dict(color=col, size=4), text=t_data_df.index[t_data_df['color']==name] if 'name' not in list(t_data_df) else t_data_df['name'][t_data_df['color']==name]))
        if G is not None:
            #pos = nx.spring_layout(G,dim=3,iterations=0,pos={i: tuple(t_data.loc[i,['x','y','z']]) for i in range(len(t_data))})
            Xed, Yed, Zed = [], [], []
            for edge in G.edges():
                if edge[0] in t_data_df.index.values and edge[1] in t_data_df.index.values:
                    Xed += [t_data_df.loc[edge[0],'x'], t_data_df.loc[edge[1],'x'], None]
                    Yed += [t_data_df.loc[edge[0],'y'], t_data_df.loc[edge[1],'y'], None]
                    Zed += [t_data_df.loc[edge[0],'z'], t_data_df.loc[edge[1],'z'], None]
            plots.append(go.Scatter3d(x=Xed,
                      y=Yed,
                      z=Zed,
                      mode='lines',
                      #line=go.scatter.Line(color='rgb(210,210,210)', width=10),
                      hoverinfo='none'
                      ))
            #print(Xed, Yed, Zed)
            #print(t_data[['x','y','z']])
    if axes_off:
        fig = go.Figure(data=plots,layout=go.Layout(scene=dict(xaxis=dict(title='',autorange=True,showgrid=False,zeroline=False,showline=False,ticks='',showticklabels=False),
            yaxis=dict(title='',autorange=True,showgrid=False,zeroline=False,showline=False,ticks='',showticklabels=False),
            zaxis=dict(title='',autorange=True,showgrid=False,zeroline=False,showline=False,ticks='',showticklabels=False))))
    else:
        fig = go.Figure(data=plots)
    py.plot(fig, filename=output_fname, auto_open=False)

def case_control_override_fn(pheno_df, column_of_interest):
    if 'case_control' in list(pheno_df):
        pheno_df.loc[pheno_df['case_control']=='normal',column_of_interest]='normal'
    return pheno_df

@visualize.command()
@click.option('-i', '--input_pkl', default='./final_preprocessed/methyl_array.pkl', help='Input database for beta and phenotype data.', type=click.Path(exists=False), show_default=True)
@click.option('-c', '--column_of_interest', default='disease', help='Column extract from phenotype data.', type=click.Path(exists=False), show_default=True)
@click.option('-o', '--output_file', default='./visualization.html', help='Output visualization.', type=click.Path(exists=False), show_default=True)
@click.option('-nn', '--n_neighbors', default=5, show_default=True, help='Number of neighbors UMAP.')
@click.option('-a', '--axes_off', is_flag=True, help='Whether to turn axes on or off.')
@click.option('-s', '--supervised', is_flag=True, help='Supervise umap embedding.')
@click.option('-d', '--min_dist', default=0.1, show_default=True, help='UMAP min distance.')
@click.option('-m', '--metric', default='euclidean', help='Reduction metric.', type=click.Choice(['euclidean','cosine']), show_default=True)
@click.option('-cc', '--case_control_override', is_flag=True, help='Add controls from case_control column and override current disease for classification tasks.', show_default=True)
def transform_plot(input_pkl, column_of_interest, output_file, n_neighbors,axes_off,supervised,min_dist, metric, case_control_override):
    """Dimensionality reduce VAE or original beta values using UMAP and plot using plotly."""
    input_dict = pickle.load(open(input_pkl,'rb'))
    try:
        input_dict['pheno'][column_of_interest]
    except:
        column_of_interest = 'disease'
    if case_control_override:
        input_dict['pheno'] = case_control_override_fn(input_dict['pheno'],column_of_interest)
    t_data = umap_embed(input_dict['beta'], input_dict['pheno'][column_of_interest], n_neighbors, supervised,min_dist, metric)
    print(t_data)
    plotly_plot(t_data, output_file, axes_off=axes_off)

@visualize.command()
@click.option('-i', '--input_csv', default='', help='Input csv.', type=click.Path(exists=False), show_default=True)
@click.option('-o', '--outfilename', default='output.png', help='Output png.', type=click.Path(exists=False), show_default=True)
@click.option('-idx', '--index_col', default=0, help='Index load dataframe', show_default=True)
@click.option('-fs', '--font_scale', default=1., help='Font scaling', show_default=True)
@click.option('-min', '--min_val', default=0., help='Min heat val', show_default=True)
@click.option('-max', '--max_val', default=1., help='Max heat val, if -1, defaults to None', show_default=True)
@click.option('-a', '--annot', is_flag=True, help='Annotate heatmap', show_default=True)
@click.option('-n', '--norm', is_flag=True, help='Normalize matrix data', show_default=True)
@click.option('-c', '--cluster', is_flag=True, help='Cluster matrix data', show_default=True)
@click.option('-m', '--matrix_type', default='none', help='Type of matrix supplied', type=click.Choice(['none','similarity','distance']), show_default=True)
@click.option('-x', '--xticks', is_flag=True, help='Show x ticks', show_default=True)
@click.option('-y', '--yticks', is_flag=True, help='Show y ticks', show_default=True)
def plot_heatmap(input_csv,outfilename,index_col,font_scale, min_val, max_val, annot,norm,cluster,matrix_type, xticks, yticks):
    import os
    os.makedirs(outfilename[:outfilename.rfind('/')],exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import seaborn as sns
    sns.set(font_scale=font_scale)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,20))
    df=pd.read_csv(input_csv,index_col=index_col)
    if norm:
        df.loc[:,:]=df.values.astype(np.float)/df.values.astype(np.float).sum(axis=1)[:, np.newaxis]
    #print(df)
    if cluster:
        import scipy.spatial as sp, scipy.cluster.hierarchy as hc
        if matrix_type=='none':
            linkage=None
        else:
            if matrix_type=='similarity':
                df = (df+df.T)/2
                print(df)
                df = 1.-df
            linkage = hc.linkage(sp.distance.squareform(df), method='average')
        sns.clustermap(df, row_linkage=linkage, col_linkage=linkage, xticklabels=xticks, yticklabels=yticks)
    else:
        sns.heatmap(df,vmin=min_val, vmax=max_val if max_val!=-1 else None, annot=annot, xticklabels=xticks, yticklabels=yticks)#,fmt='g'
    plt.tight_layout()
    plt.savefig(outfilename, dpi=300)


#################

if __name__ == '__main__':
    visualize()
