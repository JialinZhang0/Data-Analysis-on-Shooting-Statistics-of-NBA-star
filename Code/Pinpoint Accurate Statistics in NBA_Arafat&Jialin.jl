### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ 6d8bd524-ecf0-44cc-8d53-07ac59f4e5d8
begin
	import Pkg
	Pkg.add("CSV")
	Pkg.add("DataFrames")
	Pkg.add("Plots")
	Pkg.add("Clustering")
	Pkg.add("Statistics")
	Pkg.add("LinearAlgebra")
	Pkg.add("GaussianMixtures")
	Pkg.add("Distributions")
	Pkg.add("FillArrays")
	Pkg.add("HypothesisTests")
end

# ╔═╡ 89360bb8-c96a-11ec-1200-4195ac48a477
begin
	using CSV
	using DataFrames
	using Plots
	using Clustering
	using Statistics
	using LinearAlgebra
	using GaussianMixtures
	import Distributions as di
	using Random
	using Distributions
	using FillArrays
	using HypothesisTests
end

# ╔═╡ 13c3a6ac-08d0-4a79-acc9-5516400ff696
#In this project, we wanted to find Stephen Curry's 3 best shooting positions using his 2017 season data. We first plot the coordinate's of each of his shots then use kmeans clustering to localize regions where he takes shots. Using this, we were able to find his best shooting positions. We then made a Gaussian Mixture Model with the clustering data. 

#In the second portion, we analyze Stephen Curry's Playoff Data with his seasonal data to see whether or not he performs better or worse during the Playoffs. 

# ╔═╡ e908a483-0bb1-4e27-97df-302619aab9a1
#### stephen Curry's regular season data analysis

# ╔═╡ 29e7e2a5-9f64-4a13-977a-edd1f1b3f6d8
begin
	csv_reader = CSV.File("nba_savant201939.csv")
	df_reader = DataFrame(csv_reader)
end

# ╔═╡ dc221938-38a3-463d-8f63-1a542f52d2b7
names(df_reader)

# ╔═╡ 9d6e9826-cb91-49ea-8478-33d6f1437f8c
df = df_reader[:,["name","shot_made_flag","x","y","opponent","shot_distance"]]

# ╔═╡ 63656027-d53c-4b03-8640-74e642331a6d
#plot all Curry's shot (include both 1&0)
begin
	scatter(df.x,df.y,xlabel="x",ylabel="y",color=:gray, label=false,aspect_ratio=:equal)	
end

# ╔═╡ dc4116b2-e73a-4239-ae89-93084e054246
#Create a dataframe of Curry's all succussful shotings, named new_df
begin
	new_df = DataFrame()
	for i in 1:size(df.shot_made_flag,1)
		if df.shot_made_flag[i] !=0
			push!(new_df, df[i,:])
		end
	end
end

# ╔═╡ 5eea57cb-d36e-433b-a60e-cc5280667fcf
new_df

# ╔═╡ 7fbdea1a-299b-4aca-8269-d2117a99d235
# plot the xy position of new_df
begin
	scatter(new_df.x,new_df.y,xlabel="x",ylabel="y",label=false,aspect_ratio=:equal)	
end

# ╔═╡ 88bdee10-a829-4a62-89b5-ddd03ec699d2
# k mean argorithm find few centers, set 8 clusters
begin
	position_matrix = zeros(2,length(new_df.x))
	for i in 1:size(new_df.x,1)
		position_matrix[1,i] = new_df.x[i]
		position_matrix[2,i] = new_df.y[i]
	end
	n_class = 8
    Rx = kmeans(position_matrix,n_class)
end

# ╔═╡ 9fb3049f-651b-44f5-9ef6-5ac086d2a3c3
Rx.centers

# ╔═╡ da8a4f92-dcc5-4a0c-bbc3-17987ae73186
#find 2 clusters which contains the minimum two # of sucessful shotings
begin
	function second_largest(numbers)
	    m1, m2 = 10000,10000
	    for x in numbers
	        if x <= m1
	            m1, m2 = x, m1
			elseif x < m2
	            m2 = x
			end
		end
		return m1,m2
	end
end

# ╔═╡ 481aa7b7-775d-4cce-a2da-0d3099bfd410
#2 minimum # of counts
	
mins = second_largest(Rx.counts)

# ╔═╡ f7048684-d4ae-4bca-b5ae-0cd5188da07b
begin
	#get the index of the two cluster
	idx_1 = findfirst(Rx.counts .== mins[1])
	idx_2 = findfirst(Rx.counts .== mins[2])
end

# ╔═╡ f92ffbf0-75c4-493c-accf-a90de4ef173d
# plot the position and kmean centers (red are the two cluster with minimum # of successfull shots)
begin
	scatter(new_df.x,new_df.y,xlabel="x",ylabel="y",label=false,aspect_ratio=:equal)
	scatter!(Rx.centers[1,:], Rx.centers[2,:],color=:yellow) #plot k mean centers
	scatter!([Rx.centers[1,idx_1]], [Rx.centers[2,idx_1]],color=:red)
	scatter!([Rx.centers[1,idx_2]], [Rx.centers[2,idx_2]],color=:red)
end

# ╔═╡ db332c00-7953-4f30-b5ed-a0dff5a4f5f9
# pick out the dots corresponding each centers

# ╔═╡ e1e4b8fe-ef1f-4c0e-843e-85181c740a15
#use different color to show the different cluster in scatter plot
scatter(new_df.x, new_df.y, marker_z=Rx.assignments, color=:lightrainbow, legend=false)

# ╔═╡ f803d657-ffb2-4dbe-b92c-604bfd7dc446
RxAssSet = [x for x in Set(Rx.assignments)]

# ╔═╡ 85d7aebe-bae2-496f-8aee-3e04638a9b48
begin
function create_empty(n_class)
	pindex = []
	for i in 1:n_class
		new_index = []
		push!(pindex,new_index)
	end
	return pindex
end
end

# ╔═╡ cbc515e2-b416-4f1d-ab75-13eef1891e80
#got index of points in different classes
begin
	function get_pindex(pindex)
	for j in 1:n_class
		for i in 1:size(Rx.assignments,1)
			if Rx.assignments[i] == RxAssSet[j]
				push!(pindex[j],i)
			end
		end 
	end
		return pindex
	end
end

# ╔═╡ 84b895d9-3f91-4317-af84-25bf54cb375c
#pick out points corresponding to each cluster

# ╔═╡ d63d8891-93bd-405e-a196-0ca2f92d72dd
begin
	function cluster_separate(pindex,cluster,data,RxAssSet)
	for i in 1:size(pindex,1)
		for j in pindex[i]
			# push!(new_p,j)
			push!(cluster[RxAssSet[i]], data[j])
			
		end		
		#push!(cluster[i],new_p)
	end
		return cluster
	end
end	

# ╔═╡ ed0df28f-382a-42b8-a859-a95f5b3f2a3a
begin
	function check_cluster(cluster,total_num)
		count = 0
		for i in 1:size(cluster,1)
			count = length(cluster[i]) + count 
		end
		if count == total_num
			
			print("cluster correct and match!")
			return count
		else
			print("incorrect!!! ")
			return 0
		end
	end
end 

# ╔═╡ bbb76906-9d19-49d5-a9bf-3a299bc09fd6
begin
	pindex_1 = create_empty(n_class)
	pindex_1 = get_pindex(pindex_1)
	cluster_x = create_empty(n_class)
	cluster_x_new=cluster_separate(pindex_1,cluster_x,new_df.x,RxAssSet)
end

# ╔═╡ 21c0840c-95bc-4f8c-9750-01ffe78998aa
begin
	pindex_2 = create_empty(n_class)
	pindex_2 = get_pindex(pindex_2)
	cluster_y = create_empty(n_class)
	cluster_y_new=cluster_separate(pindex_2,cluster_y,new_df.y,RxAssSet)
end

# ╔═╡ 1d20216c-21b4-4396-a036-90c8bc352deb
check_cluster(cluster_x_new,length(new_df.x))

# ╔═╡ a0172bb9-b73a-437d-b0a5-45c6e11abeec
check_cluster(cluster_y_new,length(new_df.y))

# ╔═╡ 0da9ff13-f787-4798-8ea8-f9969290319e
# calculate covariance matrix and plot gaussian covariance elipse of each cluster

# ╔═╡ 2abe4d65-ed8e-4032-b146-5ba989ef724b
begin
	function cov_mat(x,y)
		new_matrix = zeros(2,2)
		new_matrix[1,1] = cov(x, x)
		new_matrix[2,2] = cov(y, y)
		new_matrix[1,2] = cov(x, y)
		new_matrix[2,1] = cov(y, x)
		return new_matrix
	end
	
end

# ╔═╡ 638addd4-4ad8-415e-b61e-7ff76af7876b
 P= cov_mat(cluster_x_new[3], cluster_y_new[3])

# ╔═╡ bcb40ded-cfb1-44c9-bd00-820bdfab0db1
U,s,_= svd(P)

# ╔═╡ 1e081f04-978f-4247-9982-fffee863f765
begin
	function covariance_ellipse(P)
		U,s,_= svd(P)
		width = sqrt(s[1])
		height = sqrt(s[2])
		if height > width
			print("width must be greater than height")
		end
		return width, height
	end
end

# ╔═╡ 81636188-c74a-4d6d-8eb1-a577a8c4fbdc
#w, h = covariance_ellipse(P)

# ╔═╡ 5c3c75cb-5dac-42da-bd14-b3bf027bfc5b
begin
	function plot_ellipse(posx, posy, w, h)
		rng = range(0, 2π, length = 221)
		ellipse(posx,posy, w, h) = Shape(w*sin.(rng).+posx, h*cos.(rng).+posy)
		elps = ellipse(posx,posy, w, h)
		plot!(elps, fillalpha = 0.2)
	end
end

# ╔═╡ 1a7d460c-fbaf-467d-850c-432928c4f9a3
begin
	function plot_cluster_ellipse(cluster_x_new, cluster_y_new, centers, index)
		P= cov_mat(cluster_x_new[index], cluster_y_new[index])
		w, h = covariance_ellipse(P)
		posx, posy = centers[:,index][1], centers[:,index][2]		
		plot_ellipse(posx, posy, w*3, h*3)
	end
end

# ╔═╡ 21a8daa9-38c7-488e-848a-752f033b637c
begin
	function get_w_h(cluster_x_new, cluster_y_new, centers, index)
		P= cov_mat(cluster_x_new[index], cluster_y_new[index])
		w, h = covariance_ellipse(P)
		return w, h
	end
end

# ╔═╡ 8fe11b61-9f3a-4885-95b0-71c072c36aae
# plot the ellipse of each cluster
begin
	for i in 1:n_class
		if i != idx_1 && i != idx_2
			plot_cluster_ellipse(cluster_x_new, cluster_y_new, Rx.centers, i)
		end
	end
	scatter!(new_df.x, new_df.y, marker_z=Rx.assignments, color=:lightrainbow, legend=false)
end

# ╔═╡ 15b85296-3305-4e70-82d6-aebb2e5af312
#calculate widths and hights of each ellipse
begin
	w_h = []
	for i in 1:n_class
			push!(w_h, get_w_h(cluster_x_new, cluster_y_new, Rx.centers, i))
	end
end

# ╔═╡ b38e994d-aac0-4346-81c9-e26eb0bcef80
#calculate the number of points in the original dataframe located in each cluster to further get the weight of successful shot in each cluster
begin	
	counts = []
	for j in 1:n_class
		count = 0
			for i in 1:size(df.x,1)				
				if (df.x[i]-Rx.centers[1,j])^2+(df.y[i]-Rx.centers[2,j])^2 <= (w_h[j][1]*3)^2+(w_h[j][2]*3)^2
					count = count+1
				end
			end
		
		push!(counts, count) 
	end
end
					

# ╔═╡ fc5b9cad-d2aa-45ee-9dd9-3d920653c869
begin
	#calculate weight of each ellipse (sucessful shot/total shots in each cluster)
	weights = []
	for i in 1:size(Rx.counts,1)
		push!(weights, Rx.counts[i]/counts[i])
	end
end

# ╔═╡ cdecd9cd-7886-4d15-aea1-8868e65c6085
#find the index of 3 centers with highest weights
begin
	first = findfirst(weights .== maximum(weights))
	second = 0
	for i in 1:size(weights,1)
		if i != first
			second = findfirst(weights .== maximum(weights[i]))
		end
	end	
	third = 0
	for i in 1:size(weights,1)
		if i != first && i != second
			third = findfirst(weights .== maximum(weights[i]))
		end
	end
end

# ╔═╡ c762c9b6-a98d-4e87-89d4-c7d292035e8c
# Using the kmeans clustering data and highest sucessful shooting rate data, we plotted the 3 best points Stephen Curry has the best chance of making in a shot. 
center_1 = Rx.centers[:,first]

# ╔═╡ 3a6b2b79-022a-41bd-acdd-ed81dcf391e6
center_2 = Rx.centers[:,second]

# ╔═╡ dc7b9b3e-d39c-40f4-a84f-d06dcee752c5
center_3 = Rx.centers[:,third]

# ╔═╡ 231ae4ba-733e-4b74-a2fb-3fa4f9e5394e
#Curry's 3 best shooting positions are denoted by the yellow star. Unsurprisingly, one of his best shooting positions is near the basketball rim. This is most likely because it is simply easier to score the closer you are to the rim. 

begin
	for i in 1:n_class
		if i != idx_1 && i != idx_2
			plot_cluster_ellipse(cluster_x_new, cluster_y_new, Rx.centers, i)
		end
	end
	scatter!(new_df.x, new_df.y, marker_z=Rx.assignments, color=:lightrainbow, legend=false)	
	scatter!([center_1[1]], [center_1[2]], color = "yellow", label = "", markershape=:star5, markersize = 10)
	scatter!([center_2[1]], [center_2[2]], color = "yellow", label = "", markershape=:star5, markersize = 10)
	scatter!([center_3[1]], [center_3[2]], color = "yellow", label = "", markershape=:star5, markersize = 10)
end

# ╔═╡ e40f9586-121c-45f5-8289-752032cace26
### the gaussian mixture model construction

# ╔═╡ 37d46146-c7b7-4fd1-a9af-786fd5feada1
begin
	P_gmm = []
	for i in 1:n_class
		if i != idx_1 && i != idx_2
			push!(P_gmm, cov_mat(cluster_x_new[i], cluster_y_new[i]))
		end
	end
end

# ╔═╡ f22dd42b-cd9b-424f-8231-d9d3dde73b24
#P_gmm

# ╔═╡ b0afe009-fe47-4020-ae77-098924afb267
# calculate percentage
begin
	function cal_per_v1(counts,total)
		percentages = []
		for i in counts
			push!(percentages,i/total)
		end
		return percentages
	end
end

# ╔═╡ 1089fb26-42ce-426b-ad01-10d8ad2b8e3a
begin
	Centers_gmm = []
	for i in 1:n_class
		if i != idx_1 && i != idx_2
			push!(Centers_gmm, [Rx.centers[1,i], Rx.centers[2,i]])
		end
	end
end

# ╔═╡ 72bc558c-8bdb-4e79-a1bd-d9afd6b7a147
begin
	# weights normalization (the sum of 8 weights is not 1, because there are some overlaps, so need to do normalization before apply to MixtureModel)
	weights_norm = []		
	for i in 1:size(weights,1)	
		weights_new = 0
		weights_new = weights[i]/sum(weights)	
		push!(weights_norm, weights_new)
	end

end

# ╔═╡ 678ff423-c04b-4cf1-a2b0-705942d96d36
# This is a gaussian mixture model we generated with our clusters
GMM = MixtureModel([di.MvNormal(Rx.centers[:,1],[std(cluster_x_new[1]), std(cluster_y_new[1])]), di.MvNormal(Rx.centers[:,2],[std(cluster_x_new[2]), std(cluster_y_new[2])]), di.MvNormal(Rx.centers[:,3],[std(cluster_x_new[3]), std(cluster_y_new[3])]), di.MvNormal(Rx.centers[:,4],[std(cluster_x_new[4]), std(cluster_y_new[4])]), di.MvNormal(Rx.centers[:,5],[std(cluster_x_new[5]), std(cluster_y_new[5])]), di.MvNormal(Rx.centers[:,6],[std(cluster_x_new[6]), std(cluster_y_new[6])]), di.MvNormal(Rx.centers[:,7],[std(cluster_x_new[7]), std(cluster_y_new[7])]), di.MvNormal(Rx.centers[:,8],[std(cluster_x_new[8]), std(cluster_y_new[8])])], [weights_norm[1], weights_norm[2], weights_norm[3], weights_norm[4], weights_norm[5], weights_norm[6], weights_norm[7], weights_norm[8]])

# ╔═╡ 2ba7b6cb-9a5a-4b92-8c09-825f2a95eed8
#Visualization of the GMM
begin
		Z = [pdf(GMM,[i,j]) for i in -400:400, j in -400:400]
		plot(-400:400,-400:400,Z,st=:surface, color=:viridis)
end

# ╔═╡ 5ab13b0d-bcad-4996-a22b-37f86525c8b8
#future work: k means clustering has some drawback, some points might be calculated multiple times, so for the more accurate calculation/prediction, we can also use neuron network

# ╔═╡ 69ac1b5c-4f7b-4004-a380-65d7cbae90ee
bar(collect(keys(Rx.counts)), collect(values(Rx.counts)), orientation=:horizontal, yticks= :all)

# ╔═╡ 8fd276e8-8927-43cd-b61d-05b3e1050b36
# In this next segment, we will attempt to see if Stephen Curry's performs better or worse during the Playoffs as compared to during the regular season. A quick Google search will show that Stephen Curry has a 47.3% shot percentage during the regular season and a 38.5% shot percentage during the Playoffs. These results alone would indicate that Stephen Curry performs worse in the Playoffs. We wanted to investigate the validity of this by comparing Stephen Curry's 2017 seasonal shot data with his total Playoff data. To do this, we compared how Stephen Curry performed against the opponents he faced in the playoffs with how he performed when he faced those same opponents during the 2017 season.    

# ╔═╡ 00ecdaee-860a-4a2d-8924-7ea2fbe62e90
new_df.opponent

# ╔═╡ 81cfdaf8-c549-4237-8e4a-ca991603c2cc
op_sets = Set(new_df.opponent)

# ╔═╡ e4850091-df1f-4a89-aead-6c66c818491a
op_list_array = [a for a in op_sets]

# ╔═╡ bde127be-0a54-4f88-8672-1bd1897d0186
op_list_array[1]

# ╔═╡ bfbcab8c-48b8-4dc3-9782-1c114ab2795a
#Dictionary with all of Stephen Curry's shots taken against all teams
begin
	op_dict_Total = Dict{String, Int}()
	#op_dict["PS"] = 0
	#op_dict["Phil"] = 0	
	for j in 1:size(op_list_array,1)
		op_dict_Total[op_list_array[j]] = 0
		for i in 1:size(df.opponent,1)
			if df.opponent[i] == op_list_array[j]
				op_dict_Total[op_list_array[j]] += 1
			end
		end
	end
end

# ╔═╡ 5acab06a-c65e-47c6-a556-f04aa333f93a
#This code is to make a dictionary with a record of all shots that stephen curry has successfully made against each team. We will use this later to calculate his shot probability against each team. 
begin
	op_dict = Dict{String, Int}()
	#op_dict["PS"] = 0
	#op_dict["Phil"] = 0	
	for j in 1:size(op_list_array,1)
		op_dict[op_list_array[j]] = 0
		for i in 1:size(new_df.opponent,1)
			if new_df.opponent[i] == op_list_array[j]
				op_dict[op_list_array[j]] += 1
			end
		end
		
	end
end

# ╔═╡ b5e52865-46e0-4b90-a5c1-c0c157325108
op_dict

# ╔═╡ 1be1bda3-888f-4653-a39d-a361819c957d
bar(collect(keys(op_dict)), collect(values(op_dict)), orientation=:horizontal, yticks= :all)

# ╔═╡ 97e9c470-37f7-4c21-9253-172d9fbe453a
#This function will delete all teams that Stephen Curry did not face in the playoffs. The purpose of this is so that we can compare the teams that he faced in both the season and playoffs to draw a better conclusion as to whether he performs better or worse in the Playoffs. 
function delete_teams(X)
delete!(X,"Miami Heat");delete!(X,"Toronto Raptors");delete!(X,"Washington Wizards");delete!(X,"Brooklyn Nets");delete!(X,"Orlando Magic");delete!(X,"Phoenix Suns"); delete!(X,"Atlanta Hawks"); delete!(X,"Utah Jazz"); delete!(X,"Detroit Pistons"); delete!(X,"Philadelphia 76ers"); delete!(X,"New York Knicks"); delete!(X,"Chicago Bulls"); delete!(X,"Minnesota Timberwolves"); delete!(X,"Boston Celtics"); delete!(X,"Dallas Mavericks"); delete!(X,"Sacramento Kings"); delete!(X,"Los Angeles Lakers")
end

# ╔═╡ 9f062c87-7d05-4884-9f3c-90ea2f1fc5d0
begin
	delete_teams(op_dict)
	delete_teams(op_dict_Total)
end

# ╔═╡ d40461fe-d070-46ab-9b5d-5b9e421758bd
#This code takes the values in the dictionary and converts it into an [Any] array. In order to get the percent shot made against each team, the shots successfully made are divided by the total shots taken then multiplied by 100. Note that the values are being sorted to match the corresponding Playoff team for a paired t-test that will later be conducted. 
begin 
	Probability_Season = []
	for i in 1:size(collect(values(sort(op_dict))),1)
		push!(Probability_Season, collect(values(sort(op_dict)))[i] ./ collect(values(sort(op_dict_Total)))[i] .* 100)
	end
end

# ╔═╡ 1e8fdfb0-0eff-481f-a0fe-fed6ed8b1e46
Probability_Season

# ╔═╡ 29e6b06d-0d35-4720-a5dd-2ce39bc6ec53
# We will then follow the same sequence of steps for the Playoff data. Below is the CSV file for Stephen Curry's Playoff statistics. 
begin
	csv_reader_Playoff = CSV.File("nba_savant (2) .csv")
	df_reader_Playoff = DataFrame(csv_reader_Playoff)
end

# ╔═╡ 0eb233a0-428f-48db-bb3f-d991a6a74381
df_Playoff = df_reader_Playoff[:,["name","shot_made_flag_Playoff","x_Playoff","y_Playoff","opponent_Playoff","shot_distance_Playoff"]]

# ╔═╡ 4944b162-d5a0-4541-ac36-b365d9faabd9
df_Playoff.shot_made_flag_Playoff[1]

# ╔═╡ df312b42-78b2-4573-b068-872fb98879c4
begin
	new_df_Playoff = DataFrame()
	for i in 1:size(df_Playoff.shot_made_flag_Playoff,1)
		if df_Playoff.shot_made_flag_Playoff[i] !=0
			push!(new_df_Playoff, df_Playoff[i,:])
		end
	end
end

# ╔═╡ f4fcdde9-b756-4cde-83fb-3a2c785aeb8a
new_df_Playoff.opponent_Playoff

# ╔═╡ d81c6036-2697-435b-be47-317a7f9b2b9f
op_sets_Playoff = Set(new_df_Playoff.opponent_Playoff)

# ╔═╡ d323b4e0-ec1e-457f-a749-e8e6b9d216df
op_list_array_Playoff = [a for a in op_sets_Playoff]

# ╔═╡ 90054b90-2f47-4290-84ff-c70235219165
#Total shots taken against each team in the Playoffs. Note, we needed to delete the Cleveland Cavaliers from this because Stephen Curry did not face against the Cleveland Cavaliers during the 2017 season. 
begin
	op_dict_TotalPlayoff = Dict{String, Int}()
	#op_dict["PS"] = 0
	#op_dict["Phil"] = 0	
	for j in 1:size(op_list_array_Playoff,1)
		op_dict_TotalPlayoff[op_list_array_Playoff[j]] = 0
		for i in 1:size(df_Playoff.opponent_Playoff,1)
			if df_Playoff.opponent_Playoff[i] == op_list_array_Playoff[j]
				op_dict_TotalPlayoff[op_list_array_Playoff[j]] += 1
			end
		end
	end
	delete!(op_dict_TotalPlayoff, "Cleveland Cavaliers")
end

# ╔═╡ 259e4cb5-25ee-432d-8723-431885823ce0
#Dictionary with Successful shots made against each team
begin
	op_dict_Playoff = Dict{String, Int}()
	#op_dict["PS"] = 0
	#op_dict["Phil"] = 0	
	for j in 1:size(op_list_array_Playoff,1)
		op_dict_Playoff[op_list_array_Playoff[j]] = 0
		for i in 1:size(new_df_Playoff.opponent_Playoff,1)
			if new_df_Playoff.opponent_Playoff[i] == op_list_array_Playoff[j]
				op_dict_Playoff[op_list_array_Playoff[j]] += 1
			end
		end
	end
	delete!(op_dict_Playoff, "Cleveland Cavaliers")
end

# ╔═╡ 440f52be-b78a-46b1-af50-ce5489631d3b
#Stephen Curry's shot made percentage against each team in the Playoffs
begin 
	Probability_Playoff = []
	for i in 1:size(collect(values(sort(op_dict_Playoff))),1)
		push!(Probability_Playoff, collect(values(sort(op_dict_Playoff)))[i] ./ collect(values(sort(op_dict_TotalPlayoff)))[i] .* 100)
	end
end

# ╔═╡ 3d7a725f-b6d1-40ff-aea1-3ac0275b7c90
Probability_Playoff

# ╔═╡ 0c0ac356-73d4-4bf5-b3d1-7290acef4ad5
#Bar graph depicting Stephen Curry's Shot made Percentage against each team during the 2017 regular season 
bar(collect(keys(sort(op_dict))), Probability_Season, orientation=:horizontal, yticks= :all,)

# ╔═╡ 12644eaf-4f55-42c4-93f0-b205ae85db28
#Bar graph depicting Stephen Curry's Shot made Percentage against each team during the Playoffs
bar(collect(keys(sort(op_dict_Playoff))), Probability_Playoff, orientation=:horizontal, yticks= :all,)

# ╔═╡ e51dcbb2-03ee-4bb2-8cc9-4c94db538124
VecP_S = Vector{Float64}(vec(Probability_Season))

# ╔═╡ ec7f309c-6a35-4e7a-b1b1-ca62109aa08a
VecP_P = Vector{Float64}(vec(Probability_Playoff))

# ╔═╡ c58d3d51-00f0-40ec-83b6-fede0a2e5eec
#We will now conduct a paired T-test to see if there is a significant difference between the probability of making a shot during the season vs probability of making a shot against the same team in the playoff. 
OneSampleTTest(vec(VecP_S), vec(VecP_P))

# ╔═╡ ef6989c6-c9b6-43cf-94c4-beceba9d5a79
#From the T-test, the results arrived at p=0.1214 which indicated that there is no significant difference between the two data sets (p>0.05). This indicates that Stephen Curry performs no worse during the playoffs compared to his season. The discrepancy in his total shot made percentage during the regular season (47.3%) as compared to his playoff shot percentage (38.5%) could be attributed to the fact that the playoff teams are harder opponents to score against. During the regular season, weaker teams could inflate Stephen Curry's field goal percentage. When meeting playoff teams during the regular season, Stephen Curry appears to perform similarly.  

# ╔═╡ Cell order:
# ╠═13c3a6ac-08d0-4a79-acc9-5516400ff696
# ╠═6d8bd524-ecf0-44cc-8d53-07ac59f4e5d8
# ╠═89360bb8-c96a-11ec-1200-4195ac48a477
# ╠═e908a483-0bb1-4e27-97df-302619aab9a1
# ╠═29e7e2a5-9f64-4a13-977a-edd1f1b3f6d8
# ╠═dc221938-38a3-463d-8f63-1a542f52d2b7
# ╠═9d6e9826-cb91-49ea-8478-33d6f1437f8c
# ╠═63656027-d53c-4b03-8640-74e642331a6d
# ╠═dc4116b2-e73a-4239-ae89-93084e054246
# ╠═5eea57cb-d36e-433b-a60e-cc5280667fcf
# ╠═7fbdea1a-299b-4aca-8269-d2117a99d235
# ╠═88bdee10-a829-4a62-89b5-ddd03ec699d2
# ╠═9fb3049f-651b-44f5-9ef6-5ac086d2a3c3
# ╠═da8a4f92-dcc5-4a0c-bbc3-17987ae73186
# ╠═481aa7b7-775d-4cce-a2da-0d3099bfd410
# ╠═f7048684-d4ae-4bca-b5ae-0cd5188da07b
# ╠═f92ffbf0-75c4-493c-accf-a90de4ef173d
# ╠═db332c00-7953-4f30-b5ed-a0dff5a4f5f9
# ╠═e1e4b8fe-ef1f-4c0e-843e-85181c740a15
# ╠═f803d657-ffb2-4dbe-b92c-604bfd7dc446
# ╠═85d7aebe-bae2-496f-8aee-3e04638a9b48
# ╠═cbc515e2-b416-4f1d-ab75-13eef1891e80
# ╠═84b895d9-3f91-4317-af84-25bf54cb375c
# ╠═d63d8891-93bd-405e-a196-0ca2f92d72dd
# ╠═ed0df28f-382a-42b8-a859-a95f5b3f2a3a
# ╠═bbb76906-9d19-49d5-a9bf-3a299bc09fd6
# ╠═21c0840c-95bc-4f8c-9750-01ffe78998aa
# ╠═1d20216c-21b4-4396-a036-90c8bc352deb
# ╠═a0172bb9-b73a-437d-b0a5-45c6e11abeec
# ╠═0da9ff13-f787-4798-8ea8-f9969290319e
# ╠═2abe4d65-ed8e-4032-b146-5ba989ef724b
# ╠═638addd4-4ad8-415e-b61e-7ff76af7876b
# ╠═bcb40ded-cfb1-44c9-bd00-820bdfab0db1
# ╠═1e081f04-978f-4247-9982-fffee863f765
# ╠═81636188-c74a-4d6d-8eb1-a577a8c4fbdc
# ╠═5c3c75cb-5dac-42da-bd14-b3bf027bfc5b
# ╠═1a7d460c-fbaf-467d-850c-432928c4f9a3
# ╠═21a8daa9-38c7-488e-848a-752f033b637c
# ╠═8fe11b61-9f3a-4885-95b0-71c072c36aae
# ╠═15b85296-3305-4e70-82d6-aebb2e5af312
# ╠═b38e994d-aac0-4346-81c9-e26eb0bcef80
# ╠═fc5b9cad-d2aa-45ee-9dd9-3d920653c869
# ╠═cdecd9cd-7886-4d15-aea1-8868e65c6085
# ╠═c762c9b6-a98d-4e87-89d4-c7d292035e8c
# ╠═3a6b2b79-022a-41bd-acdd-ed81dcf391e6
# ╠═dc7b9b3e-d39c-40f4-a84f-d06dcee752c5
# ╠═231ae4ba-733e-4b74-a2fb-3fa4f9e5394e
# ╠═e40f9586-121c-45f5-8289-752032cace26
# ╠═37d46146-c7b7-4fd1-a9af-786fd5feada1
# ╠═f22dd42b-cd9b-424f-8231-d9d3dde73b24
# ╠═b0afe009-fe47-4020-ae77-098924afb267
# ╠═1089fb26-42ce-426b-ad01-10d8ad2b8e3a
# ╠═72bc558c-8bdb-4e79-a1bd-d9afd6b7a147
# ╠═678ff423-c04b-4cf1-a2b0-705942d96d36
# ╠═2ba7b6cb-9a5a-4b92-8c09-825f2a95eed8
# ╠═5ab13b0d-bcad-4996-a22b-37f86525c8b8
# ╠═69ac1b5c-4f7b-4004-a380-65d7cbae90ee
# ╠═8fd276e8-8927-43cd-b61d-05b3e1050b36
# ╠═00ecdaee-860a-4a2d-8924-7ea2fbe62e90
# ╠═81cfdaf8-c549-4237-8e4a-ca991603c2cc
# ╠═e4850091-df1f-4a89-aead-6c66c818491a
# ╠═bde127be-0a54-4f88-8672-1bd1897d0186
# ╠═bfbcab8c-48b8-4dc3-9782-1c114ab2795a
# ╠═5acab06a-c65e-47c6-a556-f04aa333f93a
# ╠═b5e52865-46e0-4b90-a5c1-c0c157325108
# ╠═1be1bda3-888f-4653-a39d-a361819c957d
# ╠═97e9c470-37f7-4c21-9253-172d9fbe453a
# ╠═9f062c87-7d05-4884-9f3c-90ea2f1fc5d0
# ╠═d40461fe-d070-46ab-9b5d-5b9e421758bd
# ╠═1e8fdfb0-0eff-481f-a0fe-fed6ed8b1e46
# ╠═29e6b06d-0d35-4720-a5dd-2ce39bc6ec53
# ╠═0eb233a0-428f-48db-bb3f-d991a6a74381
# ╠═4944b162-d5a0-4541-ac36-b365d9faabd9
# ╠═df312b42-78b2-4573-b068-872fb98879c4
# ╠═f4fcdde9-b756-4cde-83fb-3a2c785aeb8a
# ╠═d81c6036-2697-435b-be47-317a7f9b2b9f
# ╠═d323b4e0-ec1e-457f-a749-e8e6b9d216df
# ╠═90054b90-2f47-4290-84ff-c70235219165
# ╠═259e4cb5-25ee-432d-8723-431885823ce0
# ╠═440f52be-b78a-46b1-af50-ce5489631d3b
# ╠═3d7a725f-b6d1-40ff-aea1-3ac0275b7c90
# ╠═0c0ac356-73d4-4bf5-b3d1-7290acef4ad5
# ╠═12644eaf-4f55-42c4-93f0-b205ae85db28
# ╠═e51dcbb2-03ee-4bb2-8cc9-4c94db538124
# ╠═ec7f309c-6a35-4e7a-b1b1-ca62109aa08a
# ╠═c58d3d51-00f0-40ec-83b6-fede0a2e5eec
# ╠═ef6989c6-c9b6-43cf-94c4-beceba9d5a79
