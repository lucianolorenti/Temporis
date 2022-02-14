Search.setIndex({docnames:["examples/ExampleAircraftEngine","index","reference/dataset","reference/feature_transformation/index","reference/feature_transformation/pipeline","reference/feature_transformation/step","reference/feature_transformation/target","reference/feature_transformation/transformations","reference/feature_transformation/transformer","reference/installation","reference/iterators"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,nbsphinx:4,sphinx:56},filenames:["examples/ExampleAircraftEngine.ipynb","index.rst","reference/dataset.rst","reference/feature_transformation/index.rst","reference/feature_transformation/pipeline.rst","reference/feature_transformation/step.rst","reference/feature_transformation/target.rst","reference/feature_transformation/transformations.rst","reference/feature_transformation/transformer.rst","reference/installation.rst","reference/iterators.rst"],objects:{"temporis.dataset":[[2,0,0,"-","CMAPSS"],[2,0,0,"-","ts_dataset"]],"temporis.dataset.CMAPSS":[[2,1,1,"","prepare_train_data"]],"temporis.dataset.ts_dataset":[[2,2,1,"","AbstractTimeSeriesDataset"],[2,2,1,"","FoldedDataset"]],"temporis.dataset.ts_dataset.AbstractTimeSeriesDataset":[[2,3,1,"","categorical_features"],[2,3,1,"","common_features"],[2,3,1,"","duration"],[2,3,1,"","durations"],[2,3,1,"","get_time_series"],[2,3,1,"","map"],[2,4,1,"","n_time_series"],[2,3,1,"","number_of_samples"],[2,3,1,"","number_of_samples_of_time_series"],[2,3,1,"","numeric_features"],[2,4,1,"","shape"],[2,3,1,"","to_pandas"]],"temporis.dataset.ts_dataset.FoldedDataset":[[2,3,1,"","get_time_series"],[2,4,1,"","n_time_series"],[2,3,1,"","number_of_samples_of_time_series"],[2,3,1,"","original_indices"]],"temporis.iterators":[[10,0,0,"-","batcher"],[10,0,0,"-","iterators"]],"temporis.iterators.batcher":[[10,2,1,"","Batcher"]],"temporis.iterators.batcher.Batcher":[[10,3,1,"","allocate_batch_data"],[10,4,1,"","computed_step"],[10,3,1,"","initialize_batch"],[10,4,1,"","input_shape"],[10,4,1,"","n_features"],[10,3,1,"","new"],[10,4,1,"","output_shape"],[10,4,1,"","window_size"]],"temporis.iterators.iterators":[[10,2,1,"","AbstractSampleWeights"],[10,2,1,"","InverseToLengthWeighted"],[10,2,1,"","IterationType"],[10,2,1,"","NotWeighted"],[10,2,1,"","RULInverseWeighted"],[10,2,1,"","WindowedDatasetIterator"],[10,1,1,"","seq_to_seq_signal_generator"],[10,1,1,"","valid_sample"],[10,1,1,"","windowed_signal_generator"]],"temporis.iterators.iterators.IterationType":[[10,5,1,"","FORECAST"],[10,5,1,"","SEQ_TO_SEQ"]],"temporis.iterators.iterators.WindowedDatasetIterator":[[10,3,1,"","get_data"],[10,4,1,"","input_shape"],[10,4,1,"","n_features"],[10,4,1,"","output_shape"]],"temporis.transformation.features":[[7,0,0,"-","denoising"],[7,0,0,"-","extraction"],[7,0,0,"-","imputers"],[7,0,0,"-","outliers"],[7,0,0,"-","scalers"],[7,0,0,"-","selection"],[7,0,0,"-","transformation"]],"temporis.transformation.features.denoising":[[7,2,1,"","EWMAFilter"],[7,2,1,"","GaussianFilter"],[7,2,1,"","MeanFilter"],[7,2,1,"","MedianFilter"],[7,2,1,"","MultiDimensionalKMeans"],[7,2,1,"","OneDimensionalKMeans"],[7,2,1,"","SavitzkyGolayTransformer"]],"temporis.transformation.features.denoising.EWMAFilter":[[7,3,1,"","transform"]],"temporis.transformation.features.denoising.GaussianFilter":[[7,3,1,"","transform"]],"temporis.transformation.features.denoising.MeanFilter":[[7,3,1,"","transform"]],"temporis.transformation.features.denoising.MedianFilter":[[7,3,1,"","transform"]],"temporis.transformation.features.denoising.MultiDimensionalKMeans":[[7,3,1,"","partial_fit"],[7,3,1,"","transform"]],"temporis.transformation.features.denoising.OneDimensionalKMeans":[[7,3,1,"","partial_fit"],[7,3,1,"","transform"]],"temporis.transformation.features.denoising.SavitzkyGolayTransformer":[[7,3,1,"","transform"]],"temporis.transformation.features.extraction":[[7,2,1,"","ChangesDetector"],[7,2,1,"","ColumnWiseSum"],[7,2,1,"","Difference"],[7,2,1,"","DoG"],[7,2,1,"","EMD"],[7,2,1,"","EMDFilter"],[7,2,1,"","ExpandingStatistics"],[7,2,1,"","HashingEncodingCategorical"],[7,2,1,"","HighFrequencies"],[7,2,1,"","Interactions"],[7,2,1,"","LifeStatistics"],[7,2,1,"","LowFrequencies"],[7,2,1,"","OneHotCategorical"],[7,2,1,"","ROCKET"],[7,2,1,"","RollingStatistics"],[7,2,1,"","SampleNumber"],[7,2,1,"","SimpleEncodingCategorical"],[7,2,1,"","SlidingNonOverlappingEMD"],[7,2,1,"","SlidingNonOverlappingWaveletDecomposition"],[7,2,1,"","TimeToPreviousBinaryValue"],[7,1,1,"","rolling_kurtosis"],[7,1,1,"","wrcoef"]],"temporis.transformation.features.extraction.ChangesDetector":[[7,3,1,"","transform"]],"temporis.transformation.features.extraction.ColumnWiseSum":[[7,3,1,"","transform"]],"temporis.transformation.features.extraction.Difference":[[7,3,1,"","transform"]],"temporis.transformation.features.extraction.DoG":[[7,3,1,"","transform"]],"temporis.transformation.features.extraction.EMD":[[7,3,1,"","transform"]],"temporis.transformation.features.extraction.EMDFilter":[[7,3,1,"","transform"]],"temporis.transformation.features.extraction.ExpandingStatistics":[[7,3,1,"","fit"],[7,3,1,"","partial_fit"],[7,3,1,"","transform"]],"temporis.transformation.features.extraction.HashingEncodingCategorical":[[7,3,1,"","transform"]],"temporis.transformation.features.extraction.HighFrequencies":[[7,3,1,"","fit"],[7,3,1,"","transform"]],"temporis.transformation.features.extraction.Interactions":[[7,3,1,"","transform"]],"temporis.transformation.features.extraction.LifeStatistics":[[7,3,1,"","fit"],[7,3,1,"","partial_fit"],[7,3,1,"","transform"]],"temporis.transformation.features.extraction.LowFrequencies":[[7,3,1,"","fit"],[7,3,1,"","transform"]],"temporis.transformation.features.extraction.OneHotCategorical":[[7,3,1,"","fit"],[7,3,1,"","partial_fit"],[7,3,1,"","transform"]],"temporis.transformation.features.extraction.RollingStatistics":[[7,3,1,"","fit"],[7,3,1,"","partial_fit"],[7,3,1,"","transform"]],"temporis.transformation.features.extraction.SampleNumber":[[7,3,1,"","transform"]],"temporis.transformation.features.extraction.SimpleEncodingCategorical":[[7,3,1,"","fit"],[7,3,1,"","partial_fit"],[7,3,1,"","transform"]],"temporis.transformation.features.extraction.SlidingNonOverlappingEMD":[[7,3,1,"","transform"]],"temporis.transformation.features.extraction.SlidingNonOverlappingWaveletDecomposition":[[7,3,1,"","transform"]],"temporis.transformation.features.extraction.TimeToPreviousBinaryValue":[[7,3,1,"","time_to_previous_event"],[7,3,1,"","transform"]],"temporis.transformation.features.imputers":[[7,2,1,"","FillImputer"],[7,2,1,"","ForwardFillImputer"],[7,2,1,"","MeanImputer"],[7,2,1,"","MedianImputer"],[7,2,1,"","PerColumnImputer"],[7,2,1,"","RemoveInf"],[7,2,1,"","RobustImputer"],[7,2,1,"","RollingImputer"],[7,2,1,"","RollingMeanImputer"],[7,2,1,"","RollingMedianImputer"]],"temporis.transformation.features.imputers.FillImputer":[[7,3,1,"","transform"]],"temporis.transformation.features.imputers.ForwardFillImputer":[[7,3,1,"","transform"]],"temporis.transformation.features.imputers.MeanImputer":[[7,3,1,"","fit"],[7,3,1,"","partial_fit"],[7,3,1,"","transform"]],"temporis.transformation.features.imputers.MedianImputer":[[7,3,1,"","fit"],[7,3,1,"","partial_fit"],[7,3,1,"","transform"]],"temporis.transformation.features.imputers.PerColumnImputer":[[7,3,1,"","description"],[7,3,1,"","fit"],[7,3,1,"","partial_fit"],[7,3,1,"","transform"]],"temporis.transformation.features.imputers.RemoveInf":[[7,3,1,"","transform"]],"temporis.transformation.features.imputers.RobustImputer":[[7,3,1,"","partial_fit"],[7,3,1,"","transform"]],"temporis.transformation.features.imputers.RollingImputer":[[7,3,1,"","fit"],[7,3,1,"","partial_fit"],[7,3,1,"","transform"]],"temporis.transformation.features.outliers":[[7,2,1,"","BeyondQuartileOutlierRemover"],[7,2,1,"","EWMAOutOfRange"],[7,2,1,"","EWMAOutlierRemover"],[7,2,1,"","IQROutlierRemover"],[7,2,1,"","RollingMeanOutlierRemover"],[7,2,1,"","ZScoreOutlierRemover"]],"temporis.transformation.features.outliers.BeyondQuartileOutlierRemover":[[7,3,1,"","description"],[7,3,1,"","partial_fit"],[7,3,1,"","transform"]],"temporis.transformation.features.outliers.EWMAOutOfRange":[[7,3,1,"","fit"],[7,3,1,"","partial_fit"],[7,3,1,"","transform"]],"temporis.transformation.features.outliers.EWMAOutlierRemover":[[7,3,1,"","fit"],[7,3,1,"","transform"]],"temporis.transformation.features.outliers.IQROutlierRemover":[[7,3,1,"","description"],[7,3,1,"","fit"],[7,3,1,"","partial_fit"],[7,3,1,"","transform"]],"temporis.transformation.features.outliers.RollingMeanOutlierRemover":[[7,3,1,"","transform"]],"temporis.transformation.features.outliers.ZScoreOutlierRemover":[[7,3,1,"","fit"],[7,3,1,"","transform"]],"temporis.transformation.features.scalers":[[7,2,1,"","MinMaxScaler"],[7,2,1,"","PerCategoricalMinMaxScaler"],[7,2,1,"","RobustMinMaxScaler"],[7,2,1,"","RobustStandardScaler"],[7,2,1,"","ScaleInvRUL"],[7,2,1,"","StandardScaler"]],"temporis.transformation.features.scalers.MinMaxScaler":[[7,3,1,"","description"],[7,3,1,"","fit"],[7,3,1,"","partial_fit"],[7,3,1,"","transform"]],"temporis.transformation.features.scalers.PerCategoricalMinMaxScaler":[[7,3,1,"","partial_fit"],[7,3,1,"","transform"]],"temporis.transformation.features.scalers.RobustMinMaxScaler":[[7,3,1,"","partial_fit"],[7,3,1,"","transform"]],"temporis.transformation.features.scalers.RobustStandardScaler":[[7,3,1,"","fit"],[7,3,1,"","partial_fit"],[7,3,1,"","transform"]],"temporis.transformation.features.scalers.ScaleInvRUL":[[7,3,1,"","partial_fit"],[7,3,1,"","transform"]],"temporis.transformation.features.scalers.StandardScaler":[[7,3,1,"","fit"],[7,3,1,"","partial_fit"],[7,3,1,"","transform"]],"temporis.transformation.features.selection":[[7,2,1,"","ByNameFeatureSelector"],[7,2,1,"","DiscardByNameFeatureSelector"],[7,2,1,"","LocateFeatures"],[7,2,1,"","NullProportionSelector"],[7,2,1,"","PandasNullProportionSelector"],[7,2,1,"","PandasVarianceThreshold"]],"temporis.transformation.features.selection.ByNameFeatureSelector":[[7,3,1,"","description"],[7,3,1,"","fit"],[7,4,1,"","n_features"],[7,3,1,"","partial_fit"],[7,3,1,"","transform"]],"temporis.transformation.features.selection.DiscardByNameFeatureSelector":[[7,3,1,"","fit"],[7,4,1,"","n_features"],[7,3,1,"","transform"]],"temporis.transformation.features.selection.LocateFeatures":[[7,3,1,"","transform"]],"temporis.transformation.features.selection.NullProportionSelector":[[7,3,1,"","fit"],[7,3,1,"","transform"]],"temporis.transformation.features.selection.PandasNullProportionSelector":[[7,3,1,"","fit"],[7,3,1,"","partial_fit"],[7,3,1,"","transform"]],"temporis.transformation.features.selection.PandasVarianceThreshold":[[7,3,1,"","fit"],[7,3,1,"","partial_fit"],[7,3,1,"","transform"]],"temporis.transformation.features.transformation":[[7,2,1,"","Accumulate"],[7,2,1,"","Apply"],[7,2,1,"","Diff"],[7,2,1,"","ExpandingCentering"],[7,2,1,"","ExpandingNormalization"],[7,2,1,"","MeanCentering"],[7,2,1,"","MedianCentering"],[7,2,1,"","Scale"],[7,2,1,"","Sqrt"],[7,2,1,"","Square"],[7,2,1,"","StringConcatenate"]],"temporis.transformation.features.transformation.Accumulate":[[7,3,1,"","transform"]],"temporis.transformation.features.transformation.Apply":[[7,3,1,"","transform"]],"temporis.transformation.features.transformation.Diff":[[7,3,1,"","transform"]],"temporis.transformation.features.transformation.ExpandingCentering":[[7,3,1,"","transform"]],"temporis.transformation.features.transformation.ExpandingNormalization":[[7,3,1,"","transform"]],"temporis.transformation.features.transformation.MeanCentering":[[7,3,1,"","fit"],[7,3,1,"","partial_fit"],[7,3,1,"","transform"]],"temporis.transformation.features.transformation.MedianCentering":[[7,3,1,"","fit"],[7,3,1,"","partial_fit"],[7,3,1,"","transform"]],"temporis.transformation.features.transformation.Scale":[[7,3,1,"","transform"]],"temporis.transformation.features.transformation.Sqrt":[[7,3,1,"","transform"]],"temporis.transformation.features.transformation.Square":[[7,3,1,"","transform"]],"temporis.transformation.features.transformation.StringConcatenate":[[7,3,1,"","transform"]]},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","method","Python method"],"4":["py","property","Python property"],"5":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:method","4":"py:property","5":"py:attribute"},terms:{"0":[0,2,7],"00":0,"00000":7,"000000":0,"000000e":0,"000269":0,"001953":0,"0044":0,"004518e":0,"01":0,"02":0,"027545":0,"028232":0,"03":0,"031886":0,"04":0,"043659e":0,"046446e":0,"046536e":0,"051670e":0,"058392e":0,"060648e":0,"078355":0,"094474":0,"0p1":0,"0x7f20e015bca0":0,"0x7f210010b2e0":0,"0x7f21049efb50":0,"0x7f2104b0a580":0,"0x7f2106628c10":0,"0x7f2106804550":0,"1":[0,2,7,10],"10":0,"100":[0,7],"1000":7,"1052":0,"11":[0,7],"118497e":0,"12":0,"13":0,"136852":0,"14":0,"140":0,"141103e":0,"145231":0,"15":[0,7],"151":0,"156":0,"16":0,"17":0,"172":0,"174484":0,"18":0,"180401e":0,"186825e":0,"19":0,"192":0,"19m":0,"1st":7,"2":[0,7,10],"20":0,"2008":0,"2018":0,"2021":0,"203638e":0,"208019e":0,"21":0,"22":0,"23":0,"239132e":0,"24":0,"244486e":0,"25":[0,7],"259321e":0,"25th":7,"26":0,"264424e":0,"267":0,"2683":0,"269":0,"27":0,"28":0,"280172e":0,"281455":0,"29":0,"2s":0,"3":[0,7],"30":0,"303":0,"31":0,"3150":0,"32":0,"321406e":0,"33":0,"34":0,"35":0,"36":0,"37":0,"372103":0,"374455e":0,"381319":0,"381338":0,"381352":0,"381548":0,"387741e":0,"39":0,"393232e":0,"3rd":7,"4":0,"40":0,"41":0,"418905e":0,"42":0,"43":0,"4358":0,"44":0,"446460e":0,"4467":0,"448932":0,"45":0,"450997e":0,"4512":0,"4534":0,"4655":0,"4720":0,"481781e":0,"4999":0,"5":[0,7],"50":0,"500":7,"5273":0,"5342":0,"5382":0,"539450e":0,"545":0,"5467":0,"5534":0,"566673e":0,"567381":0,"570263e":0,"5764":0,"5857":0,"589708":0,"590898e":0,"5s":0,"6":0,"602872":0,"602890":0,"6171":0,"628500e":0,"6293":0,"63":0,"64":0,"6509":0,"659648e":0,"66":0,"6621413":7,"664198e":0,"677487e":0,"7":[0,7],"71":0,"72":0,"727052e":0,"73":0,"740057e":0,"75":7,"75th":7,"770413e":0,"788760e":0,"7945":0,"7m":0,"8":0,"80":0,"845202":0,"85":0,"864672e":0,"8730":0,"881041e":0,"897551e":0,"8m":0,"9":[0,7],"920401":0,"9364":0,"952233e":0,"9527":0,"955036e":0,"970072e":0,"99":0,"case":[0,7],"class":[0,2,3,7,10],"default":[0,2,7],"float":[2,7,10],"function":7,"import":1,"int":[2,7,10],"new":[0,7,10],"null":0,"return":[0,2,7,10],"static":10,"super":0,"true":[0,7,10],A:[0,3,7,10],As:0,For:[0,7],If:[0,7],In:0,It:1,No:0,One:[0,3],The:[0,2,3,5,7,10],There:0,These:0,To:0,_:0,_________________________________________________________________:0,__getitem__:0,__init__:0,_compute_quantil:7,a4:7,about:0,abov:[0,7],abstractlivesdataset:0,abstractsampleweight:10,abstractshuffl:10,abstracttimeseriesdataset:[0,2,10],access:3,accord:7,accumul:7,activ:0,add_last:10,address:0,aero:0,after:[0,7],afterward:0,ag:0,aircraft:0,all:[0,2],allocate_batch_data:10,allow:3,also:0,an:[0,3,7,10],analys:0,analysi:1,ani:[0,7,10],anomal:7,anoth:0,appear:0,append:0,appli:[0,7],approach:0,appropri:0,approxim:7,ar:[0,7],arg:7,arnumb:7,arrai:10,assert:7,assum:0,autoreload:0,avail:0,avx2:0,avx512f:0,ax:0,b:7,base:7,batch:[0,10],batch_siz:[0,10],batcher:1,becaus:0,been:0,befor:[3,7],beggin:0,belong:7,below:7,between:[0,7],beyond:0,beyondquartileoutlierremov:7,bin:7,binari:[0,7],bool:[2,7,10],build:5,bynamefeatureselector:[0,7],c:[0,7],cache_s:2,call:[0,7],callabl:[7,10],can:[0,7],cannot:0,capabl:0,categor:7,categori:7,categorical_featur:[2,7],caus:0,cc:0,center:7,centroid:7,chang:7,changesdetector:7,check:0,clearanc:7,clip:7,cluster:7,cmapss:[0,1],cmapssdataset:0,coef_typ:7,coeff:7,coeffici:0,column:[0,2,7],column_nam:7,columnwisesum:7,come:[0,10],commerci:0,common:[0,2],common_featur:2,competit:0,compil:0,complet:0,complex_transform:0,compon:3,compos:3,compress:0,comput:[0,2,5,7],computed_step:10,concaten:[0,2],condit:0,configur:7,consid:0,consist:[0,3],constant:0,construct:[0,7,10],constructor:10,contain:[0,7,10],contamin:0,content:2,continu:0,convolut:0,core:[0,2,7],could:0,count:7,cpu:0,cpu_feature_guard:0,creat:[0,2,10],crest:7,critic:0,cuda:0,cuda_diagnost:0,cuda_driv:0,cudart:0,cudart_stub:0,cuinit:0,cumsum:7,cumul:7,current_sampl:10,cutoff:2,cycl:0,d1:7,d2:7,d3:7,d4:7,d:[0,7,10],dag:5,damag:0,data:[0,2,3,7],dataffram:7,datafram:[0,2,7],dataset:[3,7],db1:7,decomposit:7,deep:0,def:[0,7],defin:[0,2,7],degrad:0,degre:0,deliv:0,denois:1,dens:0,dense_1:0,descript:[1,7],desir:7,detect:0,df:[0,7],dict:7,diff:7,differ:[0,3,7],digest:7,ding:0,directori:0,discard:0,discardbynamefeatureselector:7,discov:0,divers:[0,7],divid:0,dlerror:0,document:0,doe:0,dog:7,domain:7,don:0,downplai:0,driver:0,drop:0,dso_load:0,dtype:0,due:0,durat:2,dure:0,dynam:0,e:0,each:[0,2,3,7,10],earli:0,effect:0,eklund:0,element:[0,7],els:0,emd:7,emdfilt:7,empir:7,enabl:0,encod:7,encourag:0,end:0,engin:0,ensembl:0,enumer:10,epoch:0,equal:0,error:0,estim:0,everi:7,ewma:7,ewmafilt:7,ewmaoutlierremov:7,ewmaoutofrang:7,exampl:0,execut:[0,7],exist:0,expand:7,expandign:7,expandingcent:7,expandingnorm:7,expandingstatist:7,expect:0,extens:3,extract:[0,1],f:[0,7],fact:0,factor:[2,7],fail:0,fals:[0,2,7,10],fault:0,fc:0,fd001:0,featur:[0,2,7,10],feature_set1:7,feature_set2:7,features1:7,features2:7,feed:[0,3],fig:0,figsiz:0,fiit:7,file:[0,2],filenam:0,fill:7,fillimput:7,filter:7,finit:0,first:[0,7],fit:7,flag:0,flatten:[0,10],fleet:0,fma:0,foldeddataset:2,follow:[0,7],forecast:10,forward:7,forwardfillimput:[0,7],frame:[2,7],framework:0,from:[0,7,10],fun:7,func:7,functional_pip:0,functional_transform:0,further:0,futur:0,g:0,gaussianfilt:7,gener:0,get:7,get_data:[0,10],get_time_seri:[0,2],given:7,go:[0,7],goebel:0,good:0,gpu:0,group:7,groupbi:0,grow:0,gt:0,guid:0,ha:0,handl:[0,3],harder:0,hash:7,hashingencodingcategor:7,have:0,held:7,helper:10,high:3,highfrequ:7,histori:0,hold:3,host:0,hot:7,how:7,http:7,hurst:7,i:[0,2,10],idea:0,identifi:0,ieee:[0,7],ieeexplor:7,ignor:0,impuls:7,imput:[0,1],includ:0,incomplet:0,increas:7,increment:7,index:[0,1,2,7],indic:[0,2],individu:7,inf:7,inform:[0,3,7,10],initi:0,initialize_batch:10,inplac:0,input:[0,3,7],input_1:0,input_shap:[0,10],inputlay:0,instruct:0,integ:7,interact:7,interfac:2,invalid:0,invers:7,inversetolengthweight:10,involv:0,invovl:0,iqr:7,iqroutlierremov:7,iter:[1,3],iteract:7,iteration_typ:10,iterationtyp:10,its:7,j:0,jsp:7,k:0,keep:7,kei:0,kernel:0,kernel_s:7,kurtosi:[0,7],kwarg:7,label:0,lambda_:7,last:0,layer:0,lcl:7,lead:7,least:3,legend:0,len:[0,7],length:2,let:[0,7],level:[3,7],li:0,libcuda:0,libcudart:0,librari:0,life:[2,7,10],life_length:10,lifestatist:7,light:2,limit:7,line2d:0,line:0,linear:0,linearli:0,list:[0,2,7],literatur:0,live:[2,7],load:0,load_ext:0,locatefeatur:7,loess:7,lolo:0,lookback:[0,10],loss:0,lot:0,lower:7,lower_quantil:7,lowfrequ:7,lt:0,luckili:0,m:[0,7],machin:0,made:7,mae:0,magnitud:0,mai:0,main:3,make:0,mani:[0,7],manufactur:0,map:[0,2],mapss:0,margin:7,mark:0,matplotlib:0,matrix:[0,10],max:[0,7],max_depth:0,max_imf:7,max_null_proport:7,max_sample_numb:10,max_work:7,mayb:0,mean:[0,7],meancent:7,meanfilt:7,meanimput:[0,7],measur:0,median:7,mediancent:7,medianfilt:7,medianimput:7,method:[0,10],min:[0,7],min_null_proport:7,min_period:7,min_point:7,min_vari:7,minimum:7,minimun:7,minmax:7,minmaxscal:[0,7],miss:7,mode:7,model:[1,3],model_select:0,modul:[1,2,3],modular:0,most:10,multidimensionalkmean:7,multipl:0,multipli:7,multivari:0,must:0,n:[0,7],n_cluster:7,n_featur:[7,10],n_kernel:7,n_time_seri:[0,2],na:7,name:[0,7],nan:7,nasa:0,nbin:7,ndarrai:10,need:0,network:0,neural:0,next:0,nlive:0,nn:0,nois:0,non:[0,7],none:[0,2,7,10],normal:[0,7],notshuffl:[0,10],notweight:10,np:[0,7,10],null_per_lif:0,null_proport:0,null_proportion_per_lif:0,nullproportionselector:7,number:[3,7,10],number_of_sampl:2,number_of_samples_of_time_seri:[0,2],number_of_std_allow:7,numer:[0,2,7],numeric_featur:2,numpi:[0,10],nvidia:0,object:[0,3,10],obtain:[0,2,7,10],one:[0,7],oneapi:0,onedimensionalkmean:7,onednn:0,onehotcategor:7,ones:0,onli:0,op2:0,op3:0,open:0,oper:[0,5],operation_mod:0,opmod:0,opset1:0,opset2:0,opset3:0,optim:0,option:[2,7,10],optiona:7,order:[0,7],org:7,original_indic:2,other:[0,3],otherwis:7,our:0,out:7,outlier:1,output:[0,7],output_s:[0,10],output_shap:10,outsid:7,outsit:7,over:7,ox:0,pad:10,page:1,pairwis:7,panda:[2,7],pandasnullproportionselector:[0,7],pandasvariancethreshold:[0,7],param:[0,7],paramet:[2,7,10],partial_fit:7,partit:7,pd:[0,2,7],peak:[0,7],per:[0,7],percategoricalminmaxscal:7,percolumnimput:[0,7],perform:[0,7],period:0,piec:0,pip:9,pipe:0,pipelin:[1,3],platform:0,plot:0,plot_pipelin:0,plt:0,pm:[0,2,3],point:[0,7,10],posit:10,possibl:[1,7],pre:0,predefin:0,predict:10,prefer:0,prepare_train_data:2,present:[0,7],prior:0,problem:[0,3],proc:0,process:0,process_file_test:0,process_file_train:0,processing_fun:0,prognost:0,progress:2,propag:0,properli:0,properti:[0,2,7,10],proport:[0,2,7],proportion_of_l:2,proportion_to_sampl:7,propot:0,propuls:0,provid:[0,2,7],pyplot:0,pywt:7,q1:7,q2:7,q3:7,q:0,quantil:7,quantile_estim:7,quantile_rang:7,quartil:7,r:7,rais:0,rand:7,randn:7,random:7,random_st:7,randomforestregressor:0,rang:[0,7],raw:0,raw_pip:0,raw_scal:0,raw_scaled_0_sensormeasure11:0,raw_scaled_0_sensormeasure12:0,raw_scaled_0_sensormeasure13:0,raw_scaled_0_sensormeasure14:0,raw_scaled_0_sensormeasure15:0,raw_scaled_0_sensormeasure17:0,raw_scaled_0_sensormeasure20:0,raw_scaled_0_sensormeasure21:0,raw_scaled_0_sensormeasure2:0,raw_scaled_0_sensormeasure3:0,raw_scaled_0_sensormeasure4:0,raw_scaled_0_sensormeasure7:0,raw_scaled_0_sensormeasure8:0,raw_scaled_0_sensormeasure9:0,reach:0,rebuild:0,receiv:0,reflect:7,regress:0,relat:3,reliabl:0,relu:0,remain:0,remov:[0,7],removeinf:[0,7],replac:7,repres:3,respect:7,respons:0,return_mask:7,rm:7,robust:7,robustimput:7,robustminmaxscal:7,robuststandardscal:7,rocket:7,roll:[0,7],rolling_kurtosi:7,rollingimput:7,rollingmeanimput:7,rollingmeanoutlierremov:7,rollingmedianimput:7,rollingstatist:[0,7],row:[0,7],rtype:10,rul:[0,2,7],rul_column:7,rule:7,rulinverseweight:10,run:0,s1:0,s20:0,s2:0,s:[0,7],safeti:0,same:[0,7],sampl:10,sample_numb:7,sample_weight:10,samplenumb:7,samples_until_end:10,savitzkygolaytransform:7,saxena:0,scale:[0,7],scale_factor:7,scaleinvrul:7,scaler:[0,1],scaler_param:7,scatter:0,search:1,second:7,see:0,select:[0,1],self:[0,7],sensor:0,sensor_indic:0,sensormeasure10:0,sensormeasure11:0,sensormeasure12:0,sensormeasure13:0,sensormeasure14:0,sensormeasure15:0,sensormeasure16:0,sensormeasure17:0,sensormeasure18:0,sensormeasure19:0,sensormeasure1:0,sensormeasure20:0,sensormeasure21:0,sensormeasure2:0,sensormeasure3:0,sensormeasure4:0,sensormeasure5:0,sensormeasure6:0,sensormeasure7:0,sensormeasure8:0,sensormeasure9:0,separ:0,seq_to_seq:10,seq_to_seq_signal_gener:10,seri:[0,2,7],set:7,set_titl:0,set_xlabel:0,set_ylabel:0,shape:[0,2,7],share:0,she:7,should:0,show:2,show_progress:[2,10],shuffl:0,shuffler:[0,10],signal:7,signal_i:10,signal_x:10,similar:0,simon:0,simpl:[0,7],simpleencodingcategor:7,simul:0,sinc:0,singl:[0,7],size:[0,7,10],skew:7,sklearn:[0,3],slide:0,slidingnonoverlappingemd:7,slidingnonoverlappingwaveletdecomposit:7,smaller:7,snapshot:0,so:0,some:[0,7],sometim:7,sort_valu:0,space:0,span:7,specifi:7,split:7,sqrt:7,squar:7,squeez:0,stage:0,stamp:7,standard:7,standardscal:7,statist:[0,7],std:[0,7],step:[0,1,3,7,10],store:0,str:[2,7],stream_executor:0,stringconcaten:7,structur:[0,7],subclass:0,subplot:0,subsampl:7,subsample_proport:2,subset:0,substanti:0,substract:7,suitabl:0,sum:7,summari:0,sun:0,suppli:7,sw:0,sw_train:0,sw_val:0,system:0,t:[0,7],tabular:0,taken:0,target:[0,1,3,10],target_pip:0,tarnsform:0,tdigest:7,tdigest_s:7,tempori:[0,2,7,9,10],ten:7,tensorflow:0,test:[0,7],text:0,tf:0,tf_regression_dataset:0,tf_train_data:0,tf_val_data:0,th:0,than:0,them:0,therefor:7,thi:[0,7,10],three:[0,3],threshold:[0,7],time:[0,2,7],time_to_previous_ev:7,timetopreviousbinaryvalu:7,to_comput:[0,7],to_panda:2,todo:7,total:0,train:7,train_batch:0,train_dataset:0,train_iter:0,train_siz:0,train_test_split:0,train_transformed_d:0,trainabl:0,trajectori:0,transform:[2,10],transformed_d:0,transformed_pip:0,transformed_sc:0,transformed_scaled_1_sensormeasure2_kurtosi:0,transformed_scaled_1_sensormeasure2_peak:0,transformed_scaled_1_sensormeasure3_kurtosi:0,transformed_scaled_1_sensormeasure3_peak:0,transformed_scaled_1_sensormeasure4_kurtosi:0,transformed_scaled_1_sensormeasure4_peak:0,transformeddataset:10,transformeri:0,transformerstep:7,transformerx:0,translat:7,trough:[0,7],true_valu:0,ts_dataset:[0,2,10],tupl:[2,7,10],two:[0,3,7],type:[0,2,7,10],ucl:7,union:[7,10],unit:[0,7],unitnumb:0,unknown:0,until:0,up:0,upper:7,upper_quantil:7,us:[0,2,3,7,10],user:0,usual:[0,7],util:0,val_iter:0,val_loss:0,val_transformed_d:0,valid:7,valid_sampl:10,validation_data:0,validation_dataset:0,valu:[7,10],valueerror:0,var_per_lif:0,vari:0,variabl:[0,7],varianc:7,variance_inform:0,variat:0,vaulu:7,version:0,visual:1,w:0,wa:7,wai:0,want:[0,7],wavedec:7,wavelet:7,wavenam:7,we:0,wear:0,wether:7,when:[2,7],where:7,whether:2,which:[0,7,10],whose:7,wil:0,window:[0,7,10],window_s:[7,10],windowed_signal_gener:10,windoweddatasetiter:10,windowediter:10,wise:[0,7],without:7,work:0,wrcoef:7,x14:0,x:[0,7],x_t:7,x_train:0,x_val:0,xgboost:0,y:[0,7],y_pred:0,y_pred_rf:0,y_train:0,y_true:0,y_val:0,you:[0,7],your:0,zero:0,zip:0,zscoreimput:7,zscoreoutlierremov:7},titles:["Getting Started","Temporis Documentation","Dataset","Feature Transformation","Transformer pipeline","Feature transformation step","Target transformation","Transformations","Transformer","Description","Dataset iterators"],titleterms:{"do":0,"function":0,"import":0,It:0,analysi:0,api:0,batcher:[0,10],boost:0,cmapss:2,complex:0,curv:0,dataset:[0,1,2,10],denois:7,descript:9,document:1,extract:7,featur:[1,3,5],fit:0,get:[0,1],gradient:0,how:0,imput:7,indic:1,instal:[1,9],iter:[0,10],kera:0,learn:0,life:0,like:0,live:0,look:0,miss:0,model:0,more:0,number:0,outlier:7,pipelin:[0,4],possibl:0,predict:0,sampl:0,scaler:7,scikit:0,select:7,set:0,start:[0,1],step:5,tabl:1,target:6,tempori:1,train:0,transform:[0,1,3,4,5,6,7,8],valid:0,valu:0,varianc:0,visual:0,windoweddatasetiter:0}})