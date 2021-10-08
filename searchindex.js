Search.setIndex({docnames:["examples/ExampleAircraftEngine","index","reference/dataset","reference/feature_transformation/index","reference/feature_transformation/pipeline","reference/feature_transformation/step","reference/feature_transformation/target","reference/feature_transformation/transformations","reference/feature_transformation/transformer","reference/installation","reference/iterators"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,nbsphinx:3,sphinx:56},filenames:["examples/ExampleAircraftEngine.ipynb","index.rst","reference/dataset.rst","reference/feature_transformation/index.rst","reference/feature_transformation/pipeline.rst","reference/feature_transformation/step.rst","reference/feature_transformation/target.rst","reference/feature_transformation/transformations.rst","reference/feature_transformation/transformer.rst","reference/installation.rst","reference/iterators.rst"],objects:{"temporis.dataset":{CMAPSS:[2,0,0,"-"],ts_dataset:[2,0,0,"-"]},"temporis.dataset.CMAPSS":{prepare_train_data:[2,1,1,""]},"temporis.dataset.ts_dataset":{AbstractTimeSeriesDataset:[2,2,1,""],FoldedDataset:[2,2,1,""]},"temporis.dataset.ts_dataset.AbstractTimeSeriesDataset":{common_features:[2,3,1,""],duration:[2,3,1,""],durations:[2,3,1,""],get_time_series:[2,3,1,""],map:[2,3,1,""],n_time_series:[2,4,1,""],number_of_samples_of_time_series:[2,3,1,""],numeric_features:[2,3,1,""],shape:[2,4,1,""],to_pandas:[2,3,1,""]},"temporis.dataset.ts_dataset.FoldedDataset":{n_time_series:[2,4,1,""]},"temporis.iterators":{batcher:[10,0,0,"-"],iterators:[10,0,0,"-"]},"temporis.iterators.batcher":{Batcher:[10,2,1,""]},"temporis.iterators.batcher.Batcher":{"new":[10,3,1,""],allocate_batch_data:[10,3,1,""],computed_step:[10,4,1,""],initialize_batch:[10,3,1,""],input_shape:[10,4,1,""],keras:[10,3,1,""],n_features:[10,4,1,""],output_shape:[10,4,1,""],window_size:[10,4,1,""]},"temporis.iterators.iterators":{AbstractSampleWeights:[10,2,1,""],InverseToLengthWeighted:[10,2,1,""],NotWeighted:[10,2,1,""],RULInverseWeighted:[10,2,1,""],WindowedDatasetIterator:[10,2,1,""],windowed_signal_generator:[10,1,1,""]},"temporis.iterators.iterators.WindowedDatasetIterator":{n_features:[10,4,1,""]},"temporis.transformation.features":{denoising:[7,0,0,"-"],extraction:[7,0,0,"-"],imputers:[7,0,0,"-"],outliers:[7,0,0,"-"],scalers:[7,0,0,"-"],selection:[7,0,0,"-"],transformation:[7,0,0,"-"]},"temporis.transformation.features.denoising":{EWMAFilter:[7,2,1,""],MeanFilter:[7,2,1,""],MedianFilter:[7,2,1,""],MultiDimensionalKMeans:[7,2,1,""],OneDimensionalKMeans:[7,2,1,""],SavitzkyGolayTransformer:[7,2,1,""]},"temporis.transformation.features.denoising.EWMAFilter":{transform:[7,3,1,""]},"temporis.transformation.features.denoising.MeanFilter":{transform:[7,3,1,""]},"temporis.transformation.features.denoising.MedianFilter":{transform:[7,3,1,""]},"temporis.transformation.features.denoising.MultiDimensionalKMeans":{partial_fit:[7,3,1,""],transform:[7,3,1,""]},"temporis.transformation.features.denoising.OneDimensionalKMeans":{partial_fit:[7,3,1,""],transform:[7,3,1,""]},"temporis.transformation.features.denoising.SavitzkyGolayTransformer":{transform:[7,3,1,""]},"temporis.transformation.features.extraction":{ChangesDetector:[7,2,1,""],Difference:[7,2,1,""],EMD:[7,2,1,""],EMDFilter:[7,2,1,""],ExpandingStatistics:[7,2,1,""],HashingEncodingCategorical:[7,2,1,""],HighFrequencies:[7,2,1,""],Interactions:[7,2,1,""],LifeStatistics:[7,2,1,""],LowFrequencies:[7,2,1,""],OneHotCategoricalPandas:[7,2,1,""],RollingStatisticsNumba:[7,2,1,""],RollingStatisticsPandas:[7,2,1,""],SampleNumber:[7,2,1,""],SimpleEncodingCategorical:[7,2,1,""],Sum:[7,2,1,""],TimeToPreviousBinaryValue:[7,2,1,""],rolling_kurtosis:[7,1,1,""]},"temporis.transformation.features.extraction.ChangesDetector":{transform:[7,3,1,""]},"temporis.transformation.features.extraction.Difference":{transform:[7,3,1,""]},"temporis.transformation.features.extraction.EMD":{transform:[7,3,1,""]},"temporis.transformation.features.extraction.EMDFilter":{transform:[7,3,1,""]},"temporis.transformation.features.extraction.ExpandingStatistics":{fit:[7,3,1,""],partial_fit:[7,3,1,""],transform:[7,3,1,""]},"temporis.transformation.features.extraction.HashingEncodingCategorical":{transform:[7,3,1,""]},"temporis.transformation.features.extraction.HighFrequencies":{fit:[7,3,1,""],transform:[7,3,1,""]},"temporis.transformation.features.extraction.Interactions":{transform:[7,3,1,""]},"temporis.transformation.features.extraction.LifeStatistics":{fit:[7,3,1,""],partial_fit:[7,3,1,""],transform:[7,3,1,""]},"temporis.transformation.features.extraction.LowFrequencies":{fit:[7,3,1,""],transform:[7,3,1,""]},"temporis.transformation.features.extraction.OneHotCategoricalPandas":{fit:[7,3,1,""],partial_fit:[7,3,1,""],transform:[7,3,1,""]},"temporis.transformation.features.extraction.RollingStatisticsNumba":{transform:[7,3,1,""]},"temporis.transformation.features.extraction.RollingStatisticsPandas":{fit:[7,3,1,""],partial_fit:[7,3,1,""],transform:[7,3,1,""]},"temporis.transformation.features.extraction.SampleNumber":{transform:[7,3,1,""]},"temporis.transformation.features.extraction.SimpleEncodingCategorical":{fit:[7,3,1,""],partial_fit:[7,3,1,""],transform:[7,3,1,""]},"temporis.transformation.features.extraction.Sum":{transform:[7,3,1,""]},"temporis.transformation.features.extraction.TimeToPreviousBinaryValue":{time_to_previous_event:[7,3,1,""],transform:[7,3,1,""]},"temporis.transformation.features.imputers":{FillImputer:[7,2,1,""],ForwardFillImputer:[7,2,1,""],PandasMeanImputer:[7,2,1,""],PandasMedianImputer:[7,2,1,""],PandasRemoveInf:[7,2,1,""],PerColumnImputer:[7,2,1,""],RollingImputer:[7,2,1,""],RollingMeanImputer:[7,2,1,""],RollingMedianImputer:[7,2,1,""]},"temporis.transformation.features.imputers.FillImputer":{transform:[7,3,1,""]},"temporis.transformation.features.imputers.ForwardFillImputer":{transform:[7,3,1,""]},"temporis.transformation.features.imputers.PandasMeanImputer":{fit:[7,3,1,""],partial_fit:[7,3,1,""],transform:[7,3,1,""]},"temporis.transformation.features.imputers.PandasMedianImputer":{fit:[7,3,1,""],partial_fit:[7,3,1,""],transform:[7,3,1,""]},"temporis.transformation.features.imputers.PandasRemoveInf":{transform:[7,3,1,""]},"temporis.transformation.features.imputers.PerColumnImputer":{fit:[7,3,1,""],partial_fit:[7,3,1,""],transform:[7,3,1,""]},"temporis.transformation.features.imputers.RollingImputer":{fit:[7,3,1,""],partial_fit:[7,3,1,""],transform:[7,3,1,""]},"temporis.transformation.features.outliers":{EWMAOutOfRange:[7,2,1,""],EWMAOutlierRemover:[7,2,1,""],IQROutlierRemover:[7,2,1,""],RollingMeanOutlierRemover:[7,2,1,""],ZScoreOutlierRemover:[7,2,1,""]},"temporis.transformation.features.outliers.EWMAOutOfRange":{fit:[7,3,1,""],partial_fit:[7,3,1,""],transform:[7,3,1,""]},"temporis.transformation.features.outliers.EWMAOutlierRemover":{fit:[7,3,1,""],transform:[7,3,1,""]},"temporis.transformation.features.outliers.IQROutlierRemover":{fit:[7,3,1,""],partial_fit:[7,3,1,""],transform:[7,3,1,""]},"temporis.transformation.features.outliers.RollingMeanOutlierRemover":{transform:[7,3,1,""]},"temporis.transformation.features.outliers.ZScoreOutlierRemover":{fit:[7,3,1,""],transform:[7,3,1,""]},"temporis.transformation.features.scalers":{PandasMinMaxScaler:[7,2,1,""],PandasRobustScaler:[7,2,1,""],PandasStandardScaler:[7,2,1,""],ScaleInvRUL:[7,2,1,""]},"temporis.transformation.features.scalers.PandasMinMaxScaler":{fit:[7,3,1,""],partial_fit:[7,3,1,""],transform:[7,3,1,""]},"temporis.transformation.features.scalers.PandasRobustScaler":{fit:[7,3,1,""],partial_fit:[7,3,1,""],transform:[7,3,1,""]},"temporis.transformation.features.scalers.PandasStandardScaler":{fit:[7,3,1,""],partial_fit:[7,3,1,""],transform:[7,3,1,""]},"temporis.transformation.features.scalers.ScaleInvRUL":{partial_fit:[7,3,1,""],transform:[7,3,1,""]},"temporis.transformation.features.selection":{ByNameFeatureSelector:[7,2,1,""],DiscardByNameFeatureSelector:[7,2,1,""],LocateFeatures:[7,2,1,""],NullProportionSelector:[7,2,1,""],PandasNullProportionSelector:[7,2,1,""],PandasVarianceThreshold:[7,2,1,""]},"temporis.transformation.features.selection.ByNameFeatureSelector":{fit:[7,3,1,""],n_features:[7,4,1,""],partial_fit:[7,3,1,""],transform:[7,3,1,""]},"temporis.transformation.features.selection.DiscardByNameFeatureSelector":{fit:[7,3,1,""],n_features:[7,4,1,""],transform:[7,3,1,""]},"temporis.transformation.features.selection.LocateFeatures":{transform:[7,3,1,""]},"temporis.transformation.features.selection.NullProportionSelector":{fit:[7,3,1,""],transform:[7,3,1,""]},"temporis.transformation.features.selection.PandasNullProportionSelector":{fit:[7,3,1,""],partial_fit:[7,3,1,""],transform:[7,3,1,""]},"temporis.transformation.features.selection.PandasVarianceThreshold":{fit:[7,3,1,""],partial_fit:[7,3,1,""],transform:[7,3,1,""]},"temporis.transformation.features.transformation":{Accumulate:[7,2,1,""],Diff:[7,2,1,""],ExpandingCentering:[7,2,1,""],ExpandingNormalization:[7,2,1,""],MeanCentering:[7,2,1,""],Scale:[7,2,1,""],Sqrt:[7,2,1,""],Square:[7,2,1,""],StringConcatenate:[7,2,1,""]},"temporis.transformation.features.transformation.Accumulate":{transform:[7,3,1,""]},"temporis.transformation.features.transformation.Diff":{transform:[7,3,1,""]},"temporis.transformation.features.transformation.ExpandingCentering":{transform:[7,3,1,""]},"temporis.transformation.features.transformation.ExpandingNormalization":{transform:[7,3,1,""]},"temporis.transformation.features.transformation.MeanCentering":{fit:[7,3,1,""],partial_fit:[7,3,1,""],transform:[7,3,1,""]},"temporis.transformation.features.transformation.Scale":{transform:[7,3,1,""]},"temporis.transformation.features.transformation.Sqrt":{transform:[7,3,1,""]},"temporis.transformation.features.transformation.Square":{transform:[7,3,1,""]},"temporis.transformation.features.transformation.StringConcatenate":{transform:[7,3,1,""]}},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","method","Python method"],"4":["py","property","Python property"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:method","4":"py:property"},terms:{"0":[0,2,7],"00":0,"00000":0,"000000":0,"000000e":0,"00001":0,"00002":0,"00003":0,"00004":0,"00005":0,"00006":0,"00007":0,"00008":0,"00009":0,"0001":0,"00010":0,"00011":0,"00012":0,"00013":0,"00014":0,"00015":0,"00016":0,"00017":0,"00018":0,"00019":0,"00020":0,"00021":0,"00022":0,"00023":0,"00024":0,"00025":0,"000269":0,"001":0,"001953":0,"0068":0,"01":0,"016132":0,"0168":0,"02":0,"020645":0,"021458":0,"0229":0,"027545":0,"028232":0,"03":0,"0300":0,"0316":0,"031886":0,"0391":0,"0397":0,"04":0,"0416":0,"0428":0,"0448":0,"0481":0,"0549":0,"0560":0,"0564":0,"0616":0,"0647":0,"0650":0,"0695":0,"070873":0,"0709":0,"0739":0,"078355":0,"0786":0,"0839":0,"0852":0,"0856":0,"094474":0,"0959":0,"0978":0,"0985":0,"0995":0,"0p1":0,"0x7f9f3c2660a0":0,"0x7f9f3c33c460":0,"0x7f9f3c3c1d60":0,"0x7f9f3c3ed850":0,"0x7f9f3c3ed9d0":0,"0x7f9f3c3edb50":0,"0x7f9f45f63cd0":0,"0x7f9f4ba861c0":0,"0x7f9f4bb2f940":0,"0x7f9f4bb76640":0,"0x7f9f4bbf9f70":0,"1":[0,2,7,10],"10":0,"100":0,"1000":0,"101":0,"1010":0,"1060":0,"1062":0,"1075":0,"1093":0,"11":0,"110":0,"1103":0,"112":0,"113":0,"1139":0,"114":0,"116096":0,"1171":0,"1176":0,"118497e":0,"12":0,"1206":0,"1241":0,"1247":0,"125":0,"126":0,"127":0,"128":0,"1281":0,"129":0,"1293":0,"13":0,"130":0,"1308":0,"131328":0,"1349":0,"1355":0,"1358":0,"1360":0,"1362":0,"136852":0,"139":0,"14":0,"145231":0,"1459":0,"1466":0,"147":0,"149":0,"15":[0,7],"1517":0,"1544":0,"1553":0,"1563":0,"1565":0,"16":0,"160":0,"161":0,"1630":0,"1651":0,"168":0,"16m":0,"17":0,"170":0,"171":0,"172":0,"1724":0,"173":0,"174":0,"174484":0,"1760":0,"1786":0,"1797":0,"18":0,"180":0,"180401e":0,"1806":0,"1822":0,"1825":0,"1841":0,"185":0,"1858":0,"186":0,"186825e":0,"188":0,"19":0,"1901":0,"191":0,"192":0,"195":0,"196":0,"1978":0,"198":0,"1998":0,"1s":0,"2":[0,7],"20":0,"2006":0,"2008":0,"201":0,"2010":0,"2015":0,"2018":0,"203":0,"203638e":0,"204":0,"2055":0,"207":0,"208":0,"209":0,"21":0,"211":0,"2110":0,"2133":0,"214":0,"2151":0,"2152":0,"2159":0,"216":0,"216826":0,"2186":0,"22":0,"2248":0,"226":0,"2276":0,"2283":0,"23":0,"2305":0,"2321":0,"2354":0,"2355":0,"237":0,"24":0,"2435":0,"246251":0,"25":[0,7],"250":0,"254":0,"2545":0,"256":0,"2562":0,"257":0,"26":0,"2629":0,"264424e":0,"2665":0,"2689":0,"27":0,"271":0,"2736":0,"2740":0,"2745":0,"2789":0,"279":0,"28":0,"280172e":0,"281455":0,"282678":0,"2845":0,"2846":0,"2865":0,"2898":0,"29":0,"2915":0,"2922":0,"2937":0,"3":[0,7],"30":0,"3000":0,"3038":0,"304058":0,"3068":0,"31":0,"3133":0,"3146":0,"315":0,"3171":0,"32":0,"321406e":0,"3253":0,"3266":0,"33":0,"3330":0,"3395":0,"34":0,"340957":0,"344598":0,"3446":0,"3457":0,"3475":0,"3495":0,"35":0,"3500":0,"3526":0,"356063":0,"36":0,"3666":0,"3699":0,"37":0,"3706":0,"372":0,"372103":0,"374455e":0,"3747":0,"3797":0,"38":0,"3800":0,"3857":0,"39":0,"3927":0,"393592":0,"3951":0,"3957":0,"3995":0,"3m":0,"3s":0,"4":0,"40":0,"4003":0,"4019":0,"402116":0,"4048":0,"4062":0,"41":0,"4110":0,"4116":0,"412559":0,"4126":0,"416699":0,"4185":0,"418905e":0,"42":0,"420":0,"42100":0,"4220":0,"4229":0,"4238":0,"4250":0,"4339":0,"4351":0,"4357":0,"437256":0,"4383":0,"4394":0,"44":0,"4406":0,"446460e":0,"448932":0,"45":0,"450997e":0,"4579":0,"4584":0,"459264":0,"46":0,"4604":0,"4661":0,"4669":0,"4671":0,"468262":0,"4686":0,"46m":0,"47":0,"4746":0,"4757":0,"477157":0,"4779":0,"47m":0,"48":0,"481781e":0,"48m":0,"49":0,"4904":0,"4993":0,"49m":0,"5":[0,7],"50":0,"500":[0,7],"5016":0,"5100":0,"512":0,"5182":0,"51m":0,"5224":0,"5241":0,"5328":0,"534081":0,"54":0,"5401":0,"5440":0,"55":0,"5506":0,"567381":0,"56m":0,"5729":0,"5742":0,"5755":0,"58":0,"5807":0,"5830":0,"5833":0,"5859":0,"589708":0,"59":0,"590":0,"590898e":0,"5950":0,"5960":0,"5m":0,"5s":0,"6":0,"60":0,"61":0,"6124":0,"6178":0,"62":0,"6209":0,"6244":0,"6273":0,"628500e":0,"63":0,"6308":0,"6371":0,"6375":0,"64":0,"6435":0,"644295":0,"644439":0,"6481":0,"65":0,"6506":0,"6534":0,"6580":0,"659648e":0,"66":0,"6641":0,"664198e":0,"6682":0,"67":0,"670572":0,"675209":0,"68":0,"6877":0,"6908":0,"6915":0,"6932":0,"6954":0,"6955":0,"6958":0,"698453":0,"6s":0,"7":0,"70":0,"7011":0,"7053":0,"71":0,"7174":0,"7188":0,"7190":0,"72":0,"7239":0,"7266":0,"727052e":0,"73":0,"7311":0,"7357":0,"7368":0,"7386":0,"74":0,"7412":0,"7451":0,"7479":0,"7483":0,"75":[0,7],"7543":0,"7569":0,"76":0,"7606":0,"7679":0,"7687":0,"7696":0,"77":0,"7706":0,"7745":0,"7760":0,"78":0,"7811":0,"7872":0,"7873":0,"788751":0,"788760e":0,"79":0,"7907":0,"7915":0,"7929":0,"795912":0,"7m":0,"7s":0,"8":0,"8004":0,"8012":0,"8013":0,"8038":0,"8050":0,"8082":0,"8130":0,"8169":0,"82":0,"8200":0,"8221":0,"8246":0,"825324":0,"8255":0,"8281":0,"8367":0,"8370":0,"8404":0,"845202":0,"8489":0,"849":0,"85125":0,"8618":0,"8621":0,"862445":0,"8634":0,"864672e":0,"8666":0,"8735":0,"8736":0,"8821":0,"8837":0,"885864":0,"8876":0,"8893":0,"8905":0,"8928":0,"8935":0,"896":0,"897551e":0,"8981":0,"8m":0,"9":0,"9006":0,"9010":0,"9011":0,"9021":0,"9078":0,"91":0,"9101":0,"9117":0,"9157":0,"92":0,"920401":0,"920854":0,"9299":0,"9358":0,"9394":0,"9396":0,"9414":0,"9460":0,"9461":0,"9471":0,"9539":0,"9543":0,"955036e":0,"9608":0,"9632":0,"968799":0,"970072e":0,"9710":0,"975119":0,"9800":0,"980034":0,"9818":0,"9826":0,"9839":0,"9869":0,"9881":0,"99":0,"9977":0,"9s":0,"\u03c1ub":0,"\u03c1ul":0,"break":0,"case":[0,7],"class":[0,2,3,7,10],"default":[2,7,10],"float":[2,7,10],"function":[7,10],"import":1,"int":[2,7,10],"new":[0,7,10],"null":0,"return":[0,2,7,10],"static":10,"true":[0,7,10],A:[1,3,7,10],And:0,As:0,For:[0,7],If:[0,7],In:0,It:0,One:[0,3],The:[0,2,3,5,7,10],There:0,These:0,_:0,_________________________________________________________________:0,__getitem__:0,__init__:0,abil:0,about:0,abscolut:7,abstractlivesdataset:0,abstractsampleweight:10,abstractshuffl:10,abstracttimeseriesdataset:[2,10],access:3,accord:7,accordingli:0,accumul:7,activ:0,add_fit:0,add_last:10,add_vertical_lin:0,addition:0,address:0,aero:0,after:[0,7],afterward:0,ag:0,aircraft:0,all:[0,2],allocate_batch_data:10,allow:[0,3],alpha:0,alreadi:0,also:0,amount:0,an:[0,3,7],analys:0,analysi:1,ani:[0,7,10],anomal:7,anoth:0,append:0,appli:0,approach:0,approxim:0,ar:[0,7],arg:7,arrai:[0,10],arriv:0,articl:0,asset:0,assum:0,averag:0,ax:0,axessubplot:0,b:7,base:0,baselinemodel:0,batch:[0,10],batch_norm:0,batch_siz:[0,10],batcher:1,becaus:0,been:0,befor:[0,3,7],beggin:0,beghi:0,belong:[0,7],between:[0,7],beyond:0,bin:7,binari:7,bool:[2,7,10],breakag:0,build:[0,5],build_model:0,bynamefeatureselector:[0,7],c:[0,7],cache_s:0,call:7,callabl:[7,10],callback:0,cam:0,can:[0,7],capabl:0,categor:7,categori:7,caus:0,center:[0,7],centroid:7,chang:[0,7],changesdetector:7,check:0,classifi:0,clearanc:7,clip:7,cluster:7,cmapss:[0,1],cmapssdataset:0,coeffici:0,coincid:0,column:[0,2,7],column_nam:7,come:[0,10],commerci:0,common:[0,2],common_featur:2,compar:0,comparison:1,competit:0,complet:0,complex:1,complex_transform:0,compon:3,compos:3,compress:0,comput:[0,2,5,7],computed_step:10,concaten:[0,2],condit:0,connect:0,conserv:0,consid:0,consist:[0,3],constant:0,construct:[0,7,10],constructor:10,contain:[0,7,10],contamin:0,content:2,context:0,continu:0,conv1d:0,conv2d:0,conv2d_1:0,conv2d_2:0,conv2d_3:0,conv2d_4:0,convolut:0,core:[2,7],could:0,count:7,creat:[0,2,10],crest:7,cross:0,cumsum:7,cumul:7,custom:1,custommodel:0,cutoff:2,cv:0,cv_barplot_errors_wrt_rul_multiple_model:0,cv_boxplot_errors_wrt_rul_multiple_model:0,cv_regression_metr:0,cycl:0,d:[0,10],dag:5,damag:0,data:[0,2,3,7],dataffram:7,datafram:[0,2,7],dataset:[3,7],decomposit:7,decreas:0,deep:0,def:0,defin:[0,2],degrad:0,degre:0,deliv:0,denois:1,dens:0,dense_1:0,dense_2:0,descript:[1,7],detect:0,df:[0,7],dictionari:0,diff:7,differ:[0,3,7],dimens:0,ding:0,discard:0,discard_threshold:10,discardbynamefeatureselector:7,discov:0,divers:[0,7],divid:0,doc:0,document:0,domain:7,don:0,downplai:0,drop:0,dropout:0,dropout_1:0,dtype:0,due:0,durat:[0,2],durations_boxplot:0,dure:0,dynam:0,e:0,each:[0,2,3,7,10],earli:0,early_stop:0,earlystop:0,easili:0,effect:0,eklund:0,element:0,els:0,emd:7,emdfilt:7,empir:7,encod:7,encourag:0,end:0,engin:0,ensembl:0,epoch:0,equal:0,error:0,estim:0,evalu:1,evenly_spaced_point:10,everi:7,ewma:7,ewmafilt:7,ewmaoutlierremov:7,ewmaoutofrang:7,exampl:0,execut:0,expand:7,expandign:7,expandingcent:7,expandingnorm:7,expandingstatist:7,expect:0,extens:[0,3],extract:[0,1],f:[0,7],fact:0,factor:[2,7],failur:0,fals:[0,2,7],fault:0,fc:0,fc_exampl:0,fcn:0,fd001:0,featur:[0,2,7,10],feature_set1:7,feature_set2:7,features1:7,features2:7,feed:[0,3],fft_centroid:7,fft_kurtosi:7,fft_skew:7,fft_varianc:7,fig:0,figsiz:0,fiit:7,file:[0,2],filenam:0,fill:7,fillimput:7,filter:7,filter_s:0,finit:0,first:[0,7],fit:7,fit_line_not_increas:0,fittedlif:0,flatten:0,fleet:0,fold:0,foldeddataset:2,follow:[0,7],forest:0,forward:7,forwardfillimput:[0,7],frame:[2,7],frequenc:[0,7],from:[0,7,10],ft:7,fuent:0,fulli:0,func:7,functional_pip:0,functional_transform:0,further:0,futur:0,g:0,gener:0,get:7,get_time_seri:2,given:7,go:0,goebel:0,good:0,gradientboost:0,graphic:0,group:7,groupbi:0,grow:0,gt:0,guid:0,ha:0,had:0,handl:[0,3],happen:0,harder:0,harm:0,hash:7,hashingencodingcategor:7,have:0,helper:10,hidden:0,high:3,highfrequ:7,hold:[0,3],hold_out_barplot_errors_wrt_rul_multiple_model:0,hold_out_boxplot_errors_wrt_rul_multiple_model:0,home:0,hot:7,how:7,hurst:7,i:[0,2,10],idea:0,identifi:0,identitytransform:0,ieee:0,implement:[0,7],impuls:7,imput:[0,1],includ:0,incomplet:0,increas:7,increment:7,incur:0,independ:0,index:[0,1,2,7],indic:[0,2],inf:7,infinit:10,info:0,inform:[0,3,7,10],inherit:0,initi:0,initialize_batch:10,inplac:0,input:[0,3,7],input_1:0,input_shap:[0,10],inputlay:0,instead:0,integ:7,interact:7,interfac:2,intern:0,interpret:0,invalid:0,invers:7,inversetolengthweight:10,involv:0,invovl:0,iqr:7,iqroutlierremov:[0,7],issu:0,iter:[1,3],iteract:7,its:[0,7],j:0,jian:0,k:0,keep:7,kei:0,kera:10,keras_batch:10,keras_regression_batch:10,kerastrainablemodel:0,kurtosi:[0,7],kwarg:7,l2:0,label:0,lambda:0,lambda_:7,last:0,layer:0,lcl:7,learning_r:0,learningrateschedul:0,least:3,legend:0,len:[0,7],length:2,let:[0,7],level:3,li:0,lia:0,librari:0,life1:0,life2:0,life:[2,7,10],life_iter:0,life_length:10,lifedatasetiter:0,lifestatist:7,lifetim:0,light:2,limit:7,line:0,linear:0,linearli:0,list:[0,2,7],literatur:0,live:[2,7],lives_dataset:0,lives_duration_histogram:0,lives_duration_histogram_from_dur:0,livespipelin:0,locatefeatur:7,loess:7,lookback:[0,10],loss:0,lot:0,lower:7,lower_quantil:7,lowfrequ:7,lr:0,lru_gcd:0,lt:0,luciano:0,luckili:0,m:[0,7],machin:0,made:[0,7],mae:0,magnitud:0,mai:0,main:[0,3],make:0,manag:0,mani:[0,7],manufactur:0,map:2,mapss:0,margin:7,mark:0,matplotlib:0,matrix:[0,10],max:[0,7],max_null_proport:7,max_window:0,mayb:0,mcloon:0,mean:[0,7],mean_baseline_model:0,meancent:7,meanfilt:7,measur:0,median:[0,7],median_baseline_model:0,medianfilt:7,method:[0,10],min:[0,7],min_null_proport:7,min_period:7,min_point:7,min_vari:7,minimum:7,minimun:7,minmaxscal:0,minut:0,miss:7,mod:0,mode:[0,7],model:[1,3],modul:[0,1,2,3],modular:0,more:1,most:[0,10],mse:0,multidimensionalkmean:7,multipl:0,multipli:7,multivari:0,must:0,n:[0,7],n_cluster:7,n_featur:[7,10],n_filter:0,n_time_seri:2,na:7,name:[0,7],nan:7,nasa:0,nbin:7,ndarrai:10,need:0,network:0,neural:0,next:0,nlive:0,nois:0,non:[0,7],none:[0,7,10],normal:[0,7],notshuffl:10,notweight:10,np:[0,7,10],null_per_lif:0,null_proport:0,null_proportion_per_lif:0,nullproportionselector:7,numba:7,number:[3,7,10],number_of_samples_of_time_seri:2,number_of_std_allow:7,numer:[0,2,7],numeric_featur:2,numpi:[0,10],object:[0,3,10],obtain:[0,2,7,10],occur:0,one:[0,7],onedimensionalkmean:7,onehotcategoricalpanda:7,ones:0,onli:0,op2:0,op3:0,oper:[0,5],operation_mod:0,opmod:0,opset1:0,opset2:0,opset3:0,option:[2,7,10],optiona:7,order:[0,7],other:[0,3],our:0,out:0,outlier:[0,1],output:[0,7],output_s:[0,10],output_shap:10,outsid:7,outsit:7,over:[0,7],ox:0,page:1,pairwis:7,pampuri:0,panda:[0,2,7],pandasmeanimput:[0,7],pandasmedianimput:7,pandasminmaxscal:[0,7],pandasnullproportionselector:[0,7],pandasremoveinf:[0,7],pandasrobustscal:7,pandasstandardscal:7,pandastransformerwrapp:0,pandasvariancethreshold:[0,7],param:[0,7],paramet:[2,7,10],partial_fit:[0,7],particular:0,patienc:0,pd:[0,2,7],peak:[0,7],per:[0,7],percentag:0,percolumnimput:[0,7],perform:0,period:0,pessimist:0,picewis:0,picewiserul:0,piec:0,pip:9,pipe:0,pipelin:[0,1,3],plot_lif:0,plot_true_and_predict:0,plot_unexpected_break:0,plot_unexploited_lifetim:0,plt:0,pm:[0,2,3],point:[0,7,10],poropous:0,posit:10,possibl:[0,7],power:7,pre:0,pred_0:0,pred_1:0,pred_m:0,pred_n:0,predefin:0,predict:10,prefer:0,prepare_train_data:2,preprocess:0,present:[0,7],prevent:0,prior:0,problem:[0,3],process:0,process_file_test:0,process_file_train:0,processing_fun:0,prognost:0,progress:2,propag:0,properli:0,properti:[0,2,7,10],proport:[0,2,7],proportion_of_l:2,proportion_to_sampl:7,propot:0,propuls:0,provid:[0,2],ps_centroid:7,ps_kurtosi:7,ps_skew:7,ps_varianc:7,pyplot:0,pytorch:0,q1:7,q2:7,q:0,qian:0,qiao:0,quantil:7,r:0,rais:0,rand:7,randn:7,random:[0,7],randomforest:0,randomforestregressor:0,rang:[0,7],rate:0,raw:0,raw_pip:0,raw_scal:0,raw_scaled_0_sensormeasure11:0,raw_scaled_0_sensormeasure12:0,raw_scaled_0_sensormeasure13:0,raw_scaled_0_sensormeasure14:0,raw_scaled_0_sensormeasure15:0,raw_scaled_0_sensormeasure17:0,raw_scaled_0_sensormeasure20:0,raw_scaled_0_sensormeasure21:0,raw_scaled_0_sensormeasure2:0,raw_scaled_0_sensormeasure3:0,raw_scaled_0_sensormeasure4:0,raw_scaled_0_sensormeasure7:0,raw_scaled_0_sensormeasure8:0,raw_scaled_0_sensormeasure9:0,reach:0,receiv:0,recommend:0,reduc:0,reducelronplateau:0,regard:0,relat:3,reliabl:0,relu:0,remain:0,remov:0,replac:7,report:0,repres:3,resampl:0,resamplertransform:0,respect:[0,7],respons:0,restart_at_end:[0,10],result:0,results_baselin:0,return_mask:7,reult:0,risk:0,rl:0,rm:7,rmse:0,robust:[0,7],robustscal:0,roll:[0,7],rolling_kurtosi:7,rollingimput:7,rollingmeanimput:7,rollingmeanoutlierremov:7,rollingmedianimput:7,rollingstatisticsnumba:7,rollingstatisticspanda:[0,7],root_mean_squared_error:0,row:[0,7],rtype:10,rul:[0,2,7],rul_column:[0,7],rul_pm:0,rul_threshold:0,rule:7,rulinverseweight:10,run:0,s1:0,s20:0,s2:0,s:[0,7],safeti:0,same:[0,7],sampl:10,sample_numb:7,sample_weight:10,samplenumb:7,savitzkygolaytransform:7,saxena:0,scale:[0,7],scale_factor:7,scaleinvrul:7,scaler:[0,1],scatter:0,scenario:0,schedul:0,schirru:0,seaborn:0,search:1,second:7,see:0,select:[0,1],select_featur:7,select_rul:0,self:[0,7],sensor:0,sensor_indic:0,sensormeasure10:0,sensormeasure11:0,sensormeasure12:0,sensormeasure13:0,sensormeasure14:0,sensormeasure15:0,sensormeasure16:0,sensormeasure17:0,sensormeasure18:0,sensormeasure19:0,sensormeasure1:0,sensormeasure20:0,sensormeasure21:0,sensormeasure2:0,sensormeasure3:0,sensormeasure4:0,sensormeasure5:0,sensormeasure6:0,sensormeasure7:0,sensormeasure8:0,sensormeasure9:0,separ:0,sequenti:0,seri:[0,2,7],set:7,set_titl:0,set_xlabel:0,set_xlim:0,set_ylabel:0,shape:[0,2,7],she:7,should:0,show:[0,2],show_progress:2,shuffl:0,shuffler:10,signal:7,signal_i:10,signal_x:10,similar:0,simon:0,simpl:[0,7],simpleencodingcategor:7,simul:0,sinc:0,singl:[0,7],size:[0,7,10],skew:7,sklearn:[0,3],sklearn_pip:0,sklearnmodel:0,slide:0,smaller:7,sn:0,snapshot:0,some:0,space:0,specifi:[0,7],spectrum:7,split:0,split_liv:0,sqrt:7,squar:7,squeez:0,stage:0,stamp:7,state:0,statist:[0,7],std:[0,7],step:[0,1,3,7,10],store:0,str:[0,2,7],stringconcaten:7,structur:0,subclass:0,subplot:0,subsample_proport:2,subset:0,substanti:0,substract:7,suggest:0,suitabl:0,sum:7,sun:0,suppli:7,susto:0,sw:0,system:0,t:0,tabular:0,taken:0,target:[0,1,3,10],target_pip:0,task:0,temp_result:0,tempori:[2,7,9,10],ten:7,tensorflow:0,test:0,text:0,tf:0,th:0,thi:[0,7,10],three:[0,3],threshold:[0,7],time:[0,2,7],time_to_previous_ev:7,timetopreviousbinaryvalu:7,titl:0,to_comput:[0,7],to_panda:2,tool:0,total:0,train:7,train_batch:0,train_dataset:0,train_iter:0,trainabl:0,trajectori:0,transform:[2,10],transformed_pip:0,transformed_sc:0,transformed_scaled_1_sensormeasure2_kurtosi:0,transformed_scaled_1_sensormeasure2_peak:0,transformed_scaled_1_sensormeasure3_kurtosi:0,transformed_scaled_1_sensormeasure3_peak:0,transformed_scaled_1_sensormeasure4_kurtosi:0,transformed_scaled_1_sensormeasure4_peak:0,transformeddataset:10,transformeri:0,transformerstep:0,transformerx:0,trough:0,true_0:0,true_1:0,true_m:0,true_n:0,true_valu:0,ts_dataset:[2,10],tupl:[2,7,10],two:[0,3,7],type:[0,2,7,10],ucl:7,ul:0,unexpect:0,unexpected_break:0,unexploit:0,unexploited_lifetim:0,union:10,unit:0,unitnumb:0,unknown:0,until:0,updat:0,upper:7,upper_quantil:7,us:[0,2,3,7,10],user:0,usual:[0,7],util:0,val_batch:0,val_iter:0,val_loss:0,val_root_mean_squared_error:0,valid:7,validation_dataset:0,valu:[7,10],valueerror:0,var_per_lif:0,vari:0,variabl:[0,7],varianc:7,variance_inform:0,variat:0,vaulu:7,verbos:0,visual:1,wa:7,wai:0,want:[0,7],we:0,wear:0,were:0,wether:[7,10],when:[0,2],where:7,whether:[2,10],which:[0,7,10],whole:0,wil:0,window:[0,7,10],window_s:[7,10],windowed_signal_gener:10,windoweddatasetiter:10,windowediter:10,wise:[0,7],without:7,work:0,written:0,x14:0,x:[0,7],x_t:7,xgboost:0,xgboostmodel:0,xiang:0,xiangqiangjianqiao:0,xiangqiangjianqiaomodel:0,xlabel:0,y:[0,7],y_axis_label:0,y_pred:0,y_pred_custom:0,y_pred_fcn:0,y_pred_gb:0,y_pred_mean:0,y_pred_median:0,y_pred_rf:0,y_pred_xqjq:0,y_true:0,y_true_custom:0,y_true_fcn:0,y_true_gb:0,y_true_mean:0,y_true_median:0,y_true_rf:0,y_true_xqjq:0,ylabel:0,you:[0,7],zero:0,zip:0,zscoreimput:7,zscoreoutlierremov:[0,7]},titles:["Getting Started","Temporis Documentation","Dataset","Feature Transformation","Transformer pipeline","Feature transformation step","Target transformation","Transformations","Transformer","Description","Dataset iterators"],titleterms:{"do":0,"function":0,"import":0,A:0,analysi:0,api:0,bar:0,baselin:0,batcher:[0,10],boost:0,box:0,cmapss:2,comparison:0,complex:0,creation:0,curv:0,custom:0,dataset:[0,1,2,10],denois:7,descript:9,document:1,evalu:0,extract:7,featur:[1,3,5],fit:0,full:0,get:[0,1],gradient:0,how:0,imput:7,indic:1,instal:[1,9],iter:[0,10],kera:0,learn:0,life:0,like:0,live:0,load:0,look:0,mainten:0,metric:0,miss:0,model:0,more:0,number:0,outlier:7,pipelin:4,plot:0,predict:0,regress:0,sampl:0,save:0,scaler:7,scikit:0,select:7,set:0,start:[0,1],step:5,tabl:1,target:6,tempori:1,train:0,transform:[0,1,3,4,5,6,7,8],valid:0,valu:0,varianc:0,visual:0,windoweddatasetiter:0}})