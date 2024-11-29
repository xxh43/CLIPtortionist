
Required Packages:

numpy
pytorch
openai-clip
kaolin
trimesh
matplotlib
imageio
joblib
plotly
pandas
scikit-learn
shapely
pymeshlab
cma
scikit-image
kaleido
Cython


Then go to folder mesh_contain:
cd mesh_contain
pip install .

Finally, go to folder src and run:

python main_ours.py --category airplane --prompt 'a fighter jet'

The results will be generated in src/exps