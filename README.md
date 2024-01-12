
<script type="text/javascript">google.load("jquery", "1.3.2");</script>

<!doctype html>
<html>
<head>
	<title>DDPMinversion</title>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
  	<link href="style.css" rel="stylesheet" type="text/css">
	<meta property="og:image" content="Path to my teaser.png"/> 
	<meta property="og:title" content="An Edit Friendly DDPM Noise Space: Inversion and Manipulations" />
	<meta property="og:description" content="Paper description." />
	<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.1/jquery.min.js"></script>
	<script src="https://kit.fontawesome.com/ad96f96272.js" crossorigin="anonymous"></script>
</head>


<body>
	<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-ka7Sk0Gln4gmtz2MlQnikT1wXgYsOg+OMhuP+IlRH9sENBO0LRn5q+8nbTov4+1p" crossorigin="anonymous"></script>
	<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">

	<!-- nav bar on the top -->
	<nav id="navbar_top" class="navbar navbar-expand-lg navbar-dark bg-primary" style="margin-bottom: 15px;">
		<div class="container">
				<a class="navbar-brand" href="">DDPMinversion</a>
			<button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#main_nav">
				<span class="navbar-toggler-icon"></span>
			</button>
			<div class="collapse navbar-collapse" id="main_nav">
			<ul class="navbar-nav ms-auto">
				<li class="nav-item"><a class="nav-link" href="#comparison">Comparisons</a></li>
				<li class="nav-item"><a class="nav-link" href="#diverse">Our diverse inversions</a></li>
			    <li class="nav-item"><a class="nav-link" href="#paper">Paper</a></li> 
			</ul>
			</div> 
		</div> 
		</nav>

	<div class="container-xl">
		
		<!-- top button -->
		<button type="button" class="btn btn-danger btn-floating btn-lg" id="btn-back-to-top">
			<i class="fas fa-arrow-up"></i>
		</button>

		<!-- top -->
		<center>
			<h1 style="font-size:42px">An Edit Friendly DDPM Noise Space: <br>Inversion and Manipulations</h1>
			<table align=center class="table" style="max-width:800px; margin-bottom:0px !important;">
				<tr>
					<td align=center>
						<span style="font-size:24px"><a href="https://inbarhub.github.io/www/" target="_blank" rel="noopener noreferrer">Inbar Huberman-Spiegelglas</a></span>
					</td>
					<td align=center>
						<span style="font-size:24px"><a href="https://www.linkedin.com/in/vova-kulikov-750b2a215" target="_blank" rel="noopener noreferrer">Vladimir Kulikov</a></span>
					</td>
					<td align=center>
						<span style="font-size:24px"><a href="https://tomer.net.technion.ac.il/" target="_blank" rel="noopener noreferrer">Tomer Michaeli</a></span>
					</td>
				</tr>
			</table>
			<table align=center class="table">
				<tr>
					<td align=center>
						<p style="font-size:24px">Technion - Israel Institute of Technology</p>
					</td>
				</tr>
			</table>
			</table>
				<table align=center class="table" style="margin-bottom:20px !important; max-width:500px">
					<tr>
						<td align=center width=200px>
							<span style="font-size:24px"><a href='resources/DDPMinversion_paper.pdf' target="_blank" rel="noopener noreferrer">[Paper]</a></span>
						</td>
						<td align=center width=200px>
							<span style="font-size:24px"><a href='resources/inversion_supp.pdf' target="_blank" rel="noopener noreferrer">[Supplementary]</a></span>
						</td>
						<td align=center width=200px>
							<span style="font-size:24px"><a href='https://github.com/inbarhub/DDPM_inversion' target="_blank" rel="noopener noreferrer">[Code]</a></span>
						</td>
					</tr>
				</table>
			</table>
		</table>
		</center>

		<!-- abstract -->
		<center>
			<table class="table smallpad" width="1" style="border-spacing: 5px; border-collapse: unset;">
				<tbody>
				<!-- <tr>
					<td style="text-align: center; font-size: 22px;"><b>Training Image</b></td>
					<td style="vertical-align:middle; text-align: center; font-size: 22px;"><b>Random Samples</b></td>
				</tr> -->
				<tr>
					<td width="18.18%">
						<img alt="" id="teaser" class="round" width="100%" src="resources/teaser.jpg"/>
					<!-- </td>
					<td width="81.82%">
						<img alt="" id="teaser_part2" class="round" width="100%" src="./resources/teaser_part2.png"/>
					</td> -->
				</tr>
				</tbody>
			</table>
		<h1>Abstract</h1>
		</center>
		<p>
			Denoising diffusion probabilistic models (DDPMs) employ a sequence of white Gaussian noise samples to generate an image. 
			In analogy with GANs, those noise maps could be considered as the latent code associated with the generated image. However, this 
			native noise space does not possess a convenient structure, and is thus challenging to work with in editing tasks. 
			Here, we propose an alternative latent noise space for DDPM that enables a wide range of editing operations via simple means, 
			and present an inversion method for extracting these edit-friendly noise maps for any given image (real or synthetically 
			generated). As opposed to the native DDPM noise space, the edit-friendly noise maps do not have a standard normal distribution 
			and are not statistically independent across timesteps. However, they allow perfect reconstruction of any desired image, and 
			simple transformations on them translate into meaningful manipulations of the output image  (e.g., shifting, color edits). 
			Moreover, in text-conditional models, fixing those noise maps while changing the text prompt, modifies semantics while
			retaining structure. We illustrate how this property enables text-based editing of real images via the diverse DDPM sampling
			scheme (in contrast to the popular non-diverse DDIM inversion). We also show how it can be used within existing
			diffusion-based editing methods to improve their quality and diversity. 
		</p>
		<!-- <center><h1>Talk</h1></center>
		<p align="center">
			<iframe width="560" height="315" src="https://www.youtube.com/embed/vKLZchVpT2E" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
		</p> -->

		<!-- style -->
		<hr id="comparison">
		<center><h1 style="margin-top: 65px;">Text Guided Editing of Real Images</h1></center>
		<!-- text guided style with retarget-->
		<center><p>Our inversion can be used for text-based editing of real images, either by itself or in combination with other editing methods.</p></center>
		
		<!-- <center>
			<table class="table smallpad" width="1" style="max-width: 1000px;">
				<tbody>
				<tr>
					<td style="text-align: center; font-size: 22px;"><b>Training Image</b></td>
					<td colspan=3 style="text-align: center; vertical-align: middle; font-size: 22px;"><b>"Monet Style"</b></td>
				</tr>
				<tr>
					<td width="24%"><img id="sunset_r_m" class="round" width="100%" src="./resources/data/sunset.png" style="max-width: 250px;"/></td>
					<td width="0.5%"></td>
					<td width="36%"><img alt="" id="sunset_r_m1" class="round" width="100%" src="./resources/style/sunset/r_monet1.png" style="max-width: 375px;"/></td>
					<td width="36%"><img alt="" id="sunset_r_m2" class="round" width="100%" src="./resources/style/sunset/r_monet2.png" style="max-width: 375px;"/></td>
				</tr>
				<tr>
					<td><img id="aurora_r_m" class="round" width="100%" src="./resources/data/aurora.png" style="max-width: 250px;"/></td>
					<td></td>
					<td><img alt="" id="aurora_r_m1" class="round" width="100%" src="./resources/style/aurora/r_monet1.png" style="max-width: 375px;"/></td>
					<td><img alt="" id="aurora_r_m2" class="round" width="100%" src="./resources/style/aurora/r_monet2.png" style="max-width: 375px;"/></td>
				</tr>
				<tr>
					<td><img id="sn_r" class="round" width="100%" src="./resources/data/starry_night.png" style="max-width: 250px;"/></td>
					<td></td>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
					<td><img alt="" id="sn_r1" class="round" width="100%" src="./resources/style/starry_night/r_monet1.png" style="max-width: 375px;"/></td>
					<td><img alt="" id="sn_r2" class="round" width="100%" src="./resources/style/starry_night/r_monet2.png" style="max-width: 375px;"/></td>
				</tr>
				<tr>
					<td style="text-align: center; font-size: 22px;"></td>
					<td colspan=3 style="text-align: center; font-size: 22px;"><b>"Van Gogh Style"</b></td>
				</tr>
				<tr>
					<td><img id="nightsky_r" class="round" width="100%" src="./resources/data/night_sky.png" style="max-width: 250px;"/></td>
					<td></td>
					<td><img alt="" id="nightsky_r1" class="round" width="100%" src="./resources/style/night_sky/r_vg1.png" style="max-width: 375px;"/></td>
					<td><img alt="" id="nightsky_r2" class="round" width="100%" src="./resources/style/night_sky/r_vg2.png" style="max-width: 375px;"/></td>
				</tr>
				<tr>
					<td><img id="sunset_r_vg" class="round" width="100%" src="./resources/data/sunset.png" style="max-width: 250px;"/></td>
					<td></td>
					<td><img alt="" id="sunset_r_vg1" class="round" width="100%" src="./resources/style/sunset/r_vg1.png" style="max-width: 375px;"/></td>
					<td><img alt="" id="sunset_r_vg2" class="round" width="100%" src="./resources/style/sunset/r_vg2.png" style="max-width: 375px;"/></td>
				</tr>
				<tr>
					<td><img id="aurora_r_vg" class="round" width="100%" src="./resources/data/aurora.png" style="max-width: 250px;"/></td>
					<td></td>
					<td><img alt="" id="aurora_r_vg1" class="round" width="100%" src="./resources/style/aurora/r_vg1.png" style="max-width: 375px;"/></td>
					<td><img alt="" id="aurora_r_vg2" class="round" width="100%" src="./resources/style/aurora/r_vg2.png" style="max-width: 375px;"/></td>
				</tr>
				</tbody>
				</table>
		</center> -->

		<!-- text guided style generation -->
		<!-- <div class="container">
			<p  style="margin-top: 35px;">
			Below are a few more examples, where the samples are in the dimensions of the training image. For each of the images below you can choose a style from the list and see a corresponding generated sample.
			</p>
		</div> -->

		<!-- dropdown -->
		<center>
		<div class="container">
			<div class="row row-cols-1 row-cols-md-1 row-cols-lg-2 row-cols-xl-2 gy-2">
				<div class="col">
					<div class="container-fluid">
						<div class="row">
							<div class="col-5">
								<div class="">
									<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px;">A photo of an <span style="color:#FF0000">old <br> church</span></h1>				
								</div>
							</div>
							<div class="col-2">
							</div>
							<div class="col-5">
								<div class="">
									<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px;">A photo of a <span style="color:#FF0000">wooden <br>house</span></h1>				
								</div>
							</div>
						</div>
						<div class="row">
							<div class="col-5">
								<div>
									<img src='resources/comparisons/church/building_4.jpg' width="100%" style="padding-right: 1%; max-width:200px;">
								</div>
							</div>
							<div class="col-2">
								<div>
									<img src='resources/arrow.png' width="100%" style="padding-right: 1%; max-height: 200px;">
									<!-- <p style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 5vw;margin-top: 40px;" width="100%" >&#8594;</p> -->
								</div>
							</div>
							<div class="col-5">
								<div>
									<img alt="" id="church" class="round" src="resources/comparisons/church/inv.png" width="100%" style="padding-right: 1%; max-width:200px; "/>
								</div>
								<div class="btn-group" id="myDropdown" style="float: right; margin-right: 7%;">
									<button class="btn btn-info">Method</button>
									<button type="button" class="btn btn-info dropdown-toggle dropdown-toggle-split" data-bs-toggle="dropdown" aria-expanded="false">
										<span class="visually-hidden">Toggle Dropdown</span>
									</button>
									<ul class="dropdown-menu">
										<li><a class="dropdown-item" onclick="document.getElementById('church').src='resources/comparisons/church/ddim.png'">DDIM inversion</a></li>
										<li><a class="dropdown-item" onclick="document.getElementById('church').src='resources/comparisons/church/pnp.png'">Plug-and-play</a></li>
										<li><a class="dropdown-item" onclick="document.getElementById('church').src='resources/comparisons/church/p2p.png'">Prompt-to-prompt</a></li>
										<li><a class="dropdown-item" onclick="document.getElementById('church').src='resources/comparisons/church/p2p_our.png'">Prompt-to-prompt with Our inversion</a></li>
										<li><a class="dropdown-item" onclick="document.getElementById('church').src='resources/comparisons/church/inv.png'">Our inversion</a></li>
										<li><hr class="dropdown-divider"></li> 
										<li><a class="dropdown-item" onclick="document.getElementById('church').src='resources/comparisons/church/building_4.jpg'">Original</a></li>
									</ul>
									<!-- <select class="">
										<option value="Our inversion" class="" onclick="document.getElementById('church').src='resources/comparisons/church/inv.png'" ></option>
										<option value="DDIM inversion" class="" onclick="document.getElementById('church').src='resources/comparisons/church/ddim.png'" ></option>
										<option class="" onclick="document.getElementById('church').src='resources/comparisons/church/p2p.png'" value="Prompt-to-prompt"></option>
										<option class="" onclick="document.getElementById('church').src='resources/comparisons/church/p2p_our.png'" value="Prompt-to-prompt with Our inversion"></option>
										<option class="" onclick="document.getElementById('church').src='resources/comparisons/church/pnp.png'" value="Plug-and-play"></option>
										<hr class="dropdown-divider"></li>
										<option class="" onclick="document.getElementById('church').src='resources/comparisons/church/building_4.jpg'" value="Original"></option>
									</select> -->
								</div>
							</div>
						</div>
					</div>
				</div>
				<div class="col">
					<div class="container-fluid">
						<div class="row">
							<div class="col-5">
								<div class="">
									<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px; "><br>A <span style="color:#FF0000">cartoon</span> of a <span style="color:#FF0000">cat</span></h1>				
								</div>
							</div>
							<div class="col-2">
							</div>
							<div class="col-5">
								<div class="">
									<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px; "><br>An <span style="color:#FF0000">image</span> of a <span style="color:#FF0000">bear</span></h1>				
								</div>
							</div>
						</div>
						<div class="row">
							<div class="col-5">
								<div>
									<img src='resources/comparisons/cat-bear/cartoon_0.jpg' width="100%" style="padding-right: 1%; max-width:200px ;">
								</div>
							</div>
							<div class="col-2">
								<div>
									<img src='resources/arrow.png' width="100%" style="padding-right: 1%; max-height: 200px;">
									<!-- <p style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 5vw;margin-top: 40px;" width="100%" >&#8594;</p> -->
								</div>
							</div>
							<div class="col-5">
								<div>
									<img alt="" id="cat-bear" class="round" width="100%" src="./resources/comparisons/cat-bear/inv.png" style="padding-right: 1%; max-width:200px; "/>
								</div>
								<div class="btn-group" id="myDropdown" style="float: right; margin-right: 7%;">
									<button type="button" style="padding-right: 1%" class="btn btn-info" >Method</button>
									<button type="button" class="btn btn-info dropdown-toggle dropdown-toggle-split" data-bs-toggle="dropdown" aria-expanded="false" >
										<span class="visually-hidden">Toggle Dropdown</span>
									</button>
									<ul class="dropdown-menu">
										<li><a class="dropdown-item" onclick="document.getElementById('cat-bear').src='./resources/comparisons/cat-bear/ddim.png'">DDIM inversion</a></li>
										<li><a class="dropdown-item" onclick="document.getElementById('cat-bear').src='./resources/comparisons/cat-bear/pnp.png'">Plug-and-play</a></li>
										<li><a class="dropdown-item" onclick="document.getElementById('cat-bear').src='./resources/comparisons/cat-bear/p2p.png'">Prompt-to-prompt</a></li>
										<li><a class="dropdown-item" onclick="document.getElementById('cat-bear').src='./resources/comparisons/cat-bear/p2p_our.png'">Prompt-to-prompt with Our inversion</a></li>
										<li><a class="dropdown-item" onclick="document.getElementById('cat-bear').src='./resources/comparisons/cat-bear/inv.png'">Our inversion</a></li>
										<li><hr class="dropdown-divider"></li> 
										<li><a class="dropdown-item" onclick="document.getElementById('cat-bear').src='./resources/comparisons/cat-bear/cartoon_0.jpg'">Original</a></li>
									</ul>
								</div>
							</div>
						</div>
					</div>
				</div>
				<div class="col">
					<div class="container-fluid">
						<div class="row">
							<div class="col-5">
								<div class="">
									<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px;">A photo of a <span style="color:#FF0000"><br>horse</span> in the field</h1>				
								</div>
							</div>
							<div class="col-2">
							</div>
							<div class="col-5">
								<div class="">
									<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px;">A photo of a <span style="color:#FF0000"><br>zebra</span> in the field</h1>				
								</div>
							</div>
						</div>
						<div class="row">
							<div class="col-5">
								<div>
									<img src='resources/comparisons/horse-zebra/horse.jpg' width="100%" style="padding-right: 1%; max-width:200px;">
								</div>
							</div>
							<div class="col-2">
								<div>
									<img src='resources/arrow.png' width="100%" style="padding-right: 1%; max-height: 200px;">
									<!-- <p style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 5vw;margin-top: 40px;" width="100%" >&#8594;</p> -->
								</div>
							</div>
							<div class="col-5">
								<div>
									<img alt="" id="horse-zebra" class="round" src="resources/comparisons/horse-zebra/inv.png" width="100%" style="padding-right: 1%; max-width:200px; "/>
								</div>
								<div class="btn-group" id="myDropdown" style="float: right; margin-right: 7%;">
									<button class="btn btn-info">Method</button>
									<button type="button" class="btn btn-info dropdown-toggle dropdown-toggle-split" data-bs-toggle="dropdown" aria-expanded="false">
										<span class="visually-hidden">Toggle Dropdown</span>
									</button>
									<ul class="dropdown-menu">
										<li><a class="dropdown-item" onclick="document.getElementById('horse-zebra').src='resources/comparisons/horse-zebra/ddim.png'">DDIM inversion</a></li>
										<li><a class="dropdown-item" onclick="document.getElementById('horse-zebra').src='resources/comparisons/horse-zebra/pnp.png'">Plug-and-play</a></li>
										<li><a class="dropdown-item" onclick="document.getElementById('horse-zebra').src='resources/comparisons/horse-zebra/p2p.png'">Prompt-to-prompt</a></li>
										<li><a class="dropdown-item" onclick="document.getElementById('horse-zebra').src='resources/comparisons/horse-zebra/p2p_our.png'">Prompt-to-prompt with Our inversion</a></li>
										<li><a class="dropdown-item" onclick="document.getElementById('horse-zebra').src='resources/comparisons/horse-zebra/inv.png'">Our inversion</a></li>
										<li><hr class="dropdown-divider"></li> 
										<li><a class="dropdown-item" onclick="document.getElementById('horse-zebra').src='resources/comparisons/horse-zebra/horse.jpg'">Original</a></li>
									</ul>
								</div>
							</div>
						</div>
					</div>
				</div>
				<div class="col">
					<div class="container-fluid">
						<div class="row">
							<div class="col-5">
								<div class="">
									<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px; ">An <span style="color:#FF0000">origami</span> of a <span style="color:#FF0000"> <br> hummingbird</span></h1>				
								</div>
							</div>
							<div class="col-2">
							</div>
							<div class="col-5">
								<div class="">
									<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px; ">A <span style="color:#FF0000">sketch</span> of a <span style="color:#FF0000"> <br>parrot</span></h1>				
								</div>
							</div>
						</div>
						<div class="row">
							<div class="col-5">
								<div>
									<img src='resources/comparisons/parrot/origami_2.jpg' width="100%" style="padding-right: 1%; max-width:200px ;">
								</div>
							</div>
							<div class="col-2">
								<div>
									<img src='resources/arrow.png' width="100%" style="padding-right: 1%; max-height: 200px;">
									<!-- <p style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 5vw;margin-top: 40px;" width="100%" >&#8594;</p> -->
								</div>
							</div>
							<div class="col-5">
								<div>
									<img alt="" id="parrot" class="round" width="100%" src="./resources/comparisons/parrot/inv.png" style="padding-right: 1%; max-width:200px; "/>
								</div>
								<div class="btn-group" id="myDropdown" style="float: right; margin-right: 7%;">
									<button type="button" style="padding-right: 1%" class="btn btn-info" >Method</button>
									<button type="button" class="btn btn-info dropdown-toggle dropdown-toggle-split" data-bs-toggle="dropdown" aria-expanded="false" >
										<span class="visually-hidden">Toggle Dropdown</span>
									</button>
									<ul class="dropdown-menu">
										<li><a class="dropdown-item" onclick="document.getElementById('parrot').src='./resources/comparisons/parrot/ddim.png'">DDIM inversion</a></li>
										<li><a class="dropdown-item" onclick="document.getElementById('parrot').src='./resources/comparisons/parrot/pnp.png'">Plug-and-play</a></li>
										<li><a class="dropdown-item" onclick="document.getElementById('parrot').src='./resources/comparisons/parrot/p2p.png'">Prompt-to-prompt</a></li>
										<li><a class="dropdown-item" onclick="document.getElementById('parrot').src='./resources/comparisons/parrot/p2p_our.png'">Prompt-to-prompt with Our inversion</a></li>
										<li><a class="dropdown-item" onclick="document.getElementById('parrot').src='./resources/comparisons/parrot/inv.png'">Our inversion</a></li>
										<li><hr class="dropdown-divider"></li> 
										<li><a class="dropdown-item" onclick="document.getElementById('parrot').src='./resources/comparisons/parrot/origami_2.jpg'">Original</a></li>
									</ul>
								</div>
							</div>
						</div>
					</div>
				</div>
				<div class="col">
					<div class="container-fluid">
						<div class="row">
							<div class="col-5">
								<div class="">
									<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px;">A scene of a <br>valley</span></h1>				
								</div>
							</div>
							<div class="col-2">
							</div>
							<div class="col-5">
								<div class="">
									<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px;">A scene of a <br>valley <span style="color:#FF0000"> with waterfall</span></h1>				
								</div>
							</div>
						</div>
						<div class="row">
							<div class="col-5">
								<div>
									<img src='resources/comparisons/valley/valley.png' width="100%" style="padding-right: 1%; max-width:200px;">
								</div>
							</div>
							<div class="col-2">
								<div>
									<img src='resources/arrow.png' width="100%" style="padding-right: 1%; max-height: 200px;">
									<!-- <p style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 5vw;margin-top: 40px;" width="100%" >&#8594;</p> -->
								</div>
							</div>
							<div class="col-5">
								<div>
									<img alt="" id="valley" class="round" src="resources/comparisons/valley/inv.png" width="100%" style="padding-right: 1%; max-width:200px; "/>
								</div>
								<div class="btn-group" id="myDropdown" style="float: right; margin-right: 7%;">
									<button class="btn btn-info">Method</button>
									<button type="button" class="btn btn-info dropdown-toggle dropdown-toggle-split" data-bs-toggle="dropdown" aria-expanded="false">
										<span class="visually-hidden">Toggle Dropdown</span>
									</button>
									<ul class="dropdown-menu">
										<li><a class="dropdown-item" onclick="document.getElementById('valley').src='resources/comparisons/valley/ddim.png'">DDIM inversion</a></li>
										<li><a class="dropdown-item" onclick="document.getElementById('valley').src='resources/comparisons/valley/pnp.png'">Plug-and-play</a></li>
										<li><a class="dropdown-item" onclick="document.getElementById('valley').src='resources/comparisons/valley/p2p.png'">Prompt-to-prompt</a></li>
										<li><a class="dropdown-item" onclick="document.getElementById('valley').src='resources/comparisons/valley/p2p_our.png'">Prompt-to-prompt with Our inversion</a></li>
										<li><a class="dropdown-item" onclick="document.getElementById('valley').src='resources/comparisons/valley/inv.png'">Our inversion</a></li>
										<li><hr class="dropdown-divider"></li> 
										<li><a class="dropdown-item" onclick="document.getElementById('valley').src='resources/comparisons/valley/valley.png'">Original</a></li>
									</ul>
								</div>
							</div>
						</div>
					</div>
				</div>
				<div class="col">
					<div class="container-fluid">
						<div class="row">
							<div class="col-5">
								<div class="">
									<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px; "><br>A <span style="color:#FF0000">toy</span> of a jeep</h1>				
								</div>
							</div>
							<div class="col-2">
							</div>
							<div class="col-5">
								<div class="">
									<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px; "><br>A <span style="color:#FF0000">cartoon</span> of a jeep</h1>				
								</div>
							</div>
						</div>
						<div class="row">
							<div class="col-5">
								<div>
									<img src='resources/comparisons/jeep/toy_18.jpg' width="100%" style="padding-right: 1%; max-width:200px ;">
								</div>
							</div>
							<div class="col-2">
								<div>
									<img src='resources/arrow.png' width="100%" style="padding-right: 1%; max-height: 200px;">
									<!-- <p style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 5vw;margin-top: 40px;" width="100%" >&#8594;</p> -->
								</div>
							</div>
							<div class="col-5">
								<div>
									<img alt="" id="jeep" class="round" width="100%" src="./resources/comparisons/jeep/inv.png" style="padding-right: 1%; max-width:200px; "/>
								</div>
								<div class="btn-group" id="myDropdown" style="float: right; margin-right: 7%;">
									<button type="button" style="padding-right: 1%" class="btn btn-info" >Method</button>
									<button type="button" class="btn btn-info dropdown-toggle dropdown-toggle-split" data-bs-toggle="dropdown" aria-expanded="false" >
										<span class="visually-hidden">Toggle Dropdown</span>
									</button>
									<ul class="dropdown-menu">
										<li><a class="dropdown-item" onclick="document.getElementById('jeep').src='./resources/comparisons/jeep/ddim.png'">DDIM inversion</a></li>
										<li><a class="dropdown-item" onclick="document.getElementById('jeep').src='./resources/comparisons/jeep/pnp.png'">Plug-and-play</a></li>
										<li><a class="dropdown-item" onclick="document.getElementById('jeep').src='./resources/comparisons/jeep/p2p.png'">Prompt-to-prompt</a></li>
										<li><a class="dropdown-item" onclick="document.getElementById('jeep').src='./resources/comparisons/jeep/p2p_our.png'">Prompt-to-prompt with Our inversion</a></li>
										<li><a class="dropdown-item" onclick="document.getElementById('jeep').src='./resources/comparisons/jeep/inv.png'">Our inversion</a></li>
										<li><hr class="dropdown-divider"></li> 
										<li><a class="dropdown-item" onclick="document.getElementById('jeep').src='./resources/comparisons/jeep/toy_18.jpg'">Original</a></li>
									</ul>
								</div>
							</div>
						</div>
					</div>
				</div>
				<!-- <div class="col" style="margin-top: 3%">
					<div class="">
						<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px; ">A photo of a <span style="color:#FF0000">horse</span> in the <span style="color:#FF0000">mud</span></h1>				
					</div>
					<div>
						<img src='resources/comparisons/horse-zebra/horse.jpg' style="padding-right: 1%; max-width:200px; ">
					</div>
				</div>
				<div class="col" style="margin-top: 6%">
					<div>
						<img src='resources/arrow.png' style="padding-right: 1%; max-height:200px;">
					</div>
				</div>
				<div class="col" style="margin-top: 3%">
					<div class="">
						<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px; right:60px">A photo of a <span style="color:#FF0000">zebra</span> in the <span style="color:#FF0000">snow</span></h1>				
					</div>
					<div>
						<img alt="" id="horse-zebra" class="round" width="100%" src="./resources/comparisons/horse-zebra/inv.png" style="padding-right: 1%; max-width:200px;"/>
					</div>
					<div class="btn-group" id="myDropdown">
						<button type="button" style="padding-right: 1%" class="btn btn-info" >Method</button>
						<button type="button" class="btn btn-info dropdown-toggle dropdown-toggle-split" data-bs-toggle="dropdown" aria-expanded="false" >
							<span class="visually-hidden">Toggle Dropdown</span>
						</button>
						<ul class="dropdown-menu">
							<li><a class="dropdown-item" onclick="document.getElementById('horse-zebra').src='./resources/comparisons/horse-zebra/ddim.png'">DDIM inversion</a></li>
							<li><a class="dropdown-item" onclick="document.getElementById('horse-zebra').src='./resources/comparisons/horse-zebra/pnp.png'">Plug-and-play</a></li>
							<li><a class="dropdown-item" onclick="document.getElementById('horse-zebra').src='./resources/comparisons/horse-zebra/p2p.png'">Prompt-to-prompt</a></li>
							<li><a class="dropdown-item" onclick="document.getElementById('horse-zebra').src='./resources/comparisons/horse-zebra/p2p_our.png'">Prompt-to-prompt with Our inversion</a></li>
							<li><a class="dropdown-item" onclick="document.getElementById('horse-zebra').src='./resources/comparisons/horse-zebra/inv.png'">Our inversion</a></li>
							<li><hr class="dropdown-divider"></li> 
							<li><a class="dropdown-item" onclick="document.getElementById('horse-zebra').src='./resources/comparisons/horse-zebra/horse.jpg'">Original</a></li>
						</ul>
					</div>
				</div>
				<div class="col" style="margin-top: 3%">
					<div class="">
						<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px; ">An <span style="color:#FF0000">origami</span> of a <span style="color:#FF0000"> <br> hummingbird</span></h1>				
					</div>
					<div>
						<img src='resources/comparisons/parrot/origami_2.jpg' style="padding-right: 1%; max-width:200px; ">
					</div>
				</div>
				<div class="col" style="margin-top: 6%">
					<div>
						<img src='resources/arrow.png' style="padding-right: 1%;max-height:200px;">
					</div>
				</div>
				<div class="col" style="margin-top: 3%">
					<div class="">
						<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px;">A <span style="color:#FF0000">sketch</span> of a <span style="color:#FF0000"> <br>parrot</span></h1>				
					</div>
					<div>
						<img alt="" id="parrot" class="round" width="100%" src="./resources/comparisons/parrot/inv.png" style="padding-right: 1%; max-width:200px; "/>
					</div>
					<div class="btn-group" id="myDropdown">
						<button type="button" style="padding-right: 1%" class="btn btn-info" >Method</button>
						<button type="button" class="btn btn-info dropdown-toggle dropdown-toggle-split" data-bs-toggle="dropdown" aria-expanded="false">
							<span class="visually-hidden">Toggle Dropdown</span>
						</button>
						<ul class="dropdown-menu">
							<li><a class="dropdown-item" onclick="document.getElementById('parrot').src='./resources/comparisons/parrot/ddim.png'">DDIM inversion</a></li>
							<li><a class="dropdown-item" onclick="document.getElementById('parrot').src='./resources/comparisons/parrot/pnp.png'">Plug-and-play</a></li>
							<li><a class="dropdown-item" onclick="document.getElementById('parrot').src='./resources/comparisons/parrot/p2p.png'">Prompt-to-prompt</a></li>
							<li><a class="dropdown-item" onclick="document.getElementById('parrot').src='./resources/comparisons/parrot/p2p_our.png'">Prompt-to-prompt with Our inversion</a></li>
							<li><a class="dropdown-item" onclick="document.getElementById('parrot').src='./resources/comparisons/parrot/inv.png'">Our inversion</a></li>
							<li><hr class="dropdown-divider"></li> 
							<li><a class="dropdown-item" onclick="document.getElementById('parrot').src='./resources/comparisons/parrot/origami_2.jpg'">Original</a></li>
						</ul>
					</div>
				</div>
				<div class="col" style="margin-top: 3%">
					<div class="">
						<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px;">A scene of a valley</h1>				
					</div>
					<div>
						<img src='resources/comparisons/valley/valley.png' style="padding-right: 1%; max-width:200px;">
					</div>
				</div>
				<div class="col" style="margin-top: 6%">
					<div>
						<img src='resources/arrow.png' style="padding-right: 1%;max-height:200px;max-width:90px;">
					</div>
				</div>
				<div class="col" style="margin-top: 3%">
					<div class="">
						<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px; bottom:18px">A scene of a valley <br> <span style="color:#FF0000">with waterfall</span></h1>				
					</div>
					<div>
						<img alt="" id="valley" class="round" width="100%" src="./resources/comparisons/valley/inv.png" style="padding-right: 1%; max-width:200px;"/>
					</div>
					<div class="btn-group" id="myDropdown">
						<button type="button" style="padding-right: 1%" class="btn btn-info" >Method</button>
						<button type="button" class="btn btn-info dropdown-toggle dropdown-toggle-split" data-bs-toggle="dropdown" aria-expanded="false" >
							<span class="visually-hidden">Toggle Dropdown</span>
						</button>
						<ul class="dropdown-menu">
							<li><a class="dropdown-item" onclick="document.getElementById('valley').src='./resources/comparisons/valley/ddim.png'; ">DDIM inversion</a></li>
							<li><a class="dropdown-item" onclick="document.getElementById('valley').src='./resources/comparisons/valley/pnp.png'">Plug-and-play</a></li>
							<li><a class="dropdown-item" onclick="document.getElementById('valley').src='./resources/comparisons/valley/p2p.png'">Prompt-to-prompt</a></li>
							<li><a class="dropdown-item" onclick="document.getElementById('valley').src='./resources/comparisons/valley/p2p_our.png'">Prompt-to-prompt with Our inversion</a></li>
							<li><a class="dropdown-item" onclick="document.getElementById('valley').src='./resources/comparisons/valley/inv.png'">Our inversion</a></li>
							<li><hr class="dropdown-divider"></li> 
							<li><a class="dropdown-item" onclick="document.getElementById('valley').src='./resources/comparisons/valley/valley.png'">Original</a></li>
						</ul>
					</div>
				</div>
				<div class="col" style="margin-top: 3%">
					<div class="">
						<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px;"><br>A <span style="color:#FF0000">toy</span> of a jeep</h1>				
					</div>
					<div>
						<img src='resources/comparisons/jeep/toy_18.jpg' style="padding-right: 1%; max-width:200px; ">
					</div>
				</div>
				<div class="col" style="margin-top: 6%">
					<div>
						<img src='resources/arrow.png' style="padding-right: 1%;max-height:200px;max-width:90px;">
					</div>
				</div>
				<div class="col" style="margin-top: 3%">
					<div class="">
						<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px; "><br>A <span style="color:#FF0000">cartoon</span> of a jeep</h1>				
					</div>
					<div>
						<img alt="" id="jeep" class="round" width="100%" src="./resources/comparisons/jeep/inv.png" style="padding-right: 1%; max-width:200px; "/>
					</div>
					<div class="btn-group" id="myDropdown">
						<button type="button" style="padding-right: 1%" class="btn btn-info" >Method</button>
						<button type="button" class="btn btn-info dropdown-toggle dropdown-toggle-split" data-bs-toggle="dropdown" aria-expanded="false">
							<span class="visually-hidden">Toggle Dropdown</span>
						</button>
						<ul class="dropdown-menu">
							<li><a class="dropdown-item" onclick="document.getElementById('jeep').src='./resources/comparisons/jeep/ddim.png'">DDIM inversion</a></li>
							<li><a class="dropdown-item" onclick="document.getElementById('jeep').src='./resources/comparisons/jeep/pnp.png'">Plug-and-play</a></li>
							<li><a class="dropdown-item" onclick="document.getElementById('jeep').src='./resources/comparisons/jeep/p2p.png'">Prompt-to-prompt</a></li>
							<li><a class="dropdown-item" onclick="document.getElementById('jeep').src='./resources/comparisons/jeep/p2p_our.png'">Prompt-to-prompt with Our inversion</a></li>
							<li><a class="dropdown-item" onclick="document.getElementById('jeep').src='./resources/comparisons/jeep/inv.png'">Our inversion</a></li>
							<li><hr class="dropdown-divider"></li> 
							<li><a class="dropdown-item" onclick="document.getElementById('jeep').src='./resources/comparisons/jeep/toy_18.jpg'">Original</a></li>
						</ul>
					<div>
				</div> -->
			</div>
		</div>
		</center>


		<!-- images diverse -->
		<hr id="diverse" style="margin-top: 6%">

		<center><h1 style="margin-top: 65px;">Diversity in Our Method</h1></center>
		<center><p>Due to the stochastic nature of our method, we can generate diverse outputs, <br> a feature that is not naturally available with
            methods relying on the DDIM inversion.</p></center>

		<center>
			<!-- 2 -->
			<div class="row row-cols-1 row-cols-md-1 row-cols-lg-2 row-cols-xl-2 gy-2">
				<div class="col">
					<div class="container-fluid">
						<div class="row">
							<div class="col-5">
								<div class="">
									<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px;">A photo of a <span style="color:#FF0000">car</span> on the side of the street</span></h1>				
								</div>
							</div>
							<div class="col-2">
							</div>
							<div class="col-5">
								<div class="">
									<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px;">A photo of a <span style="color:#FF0000">truck</span> on the side of the street</span></h1>				
								</div>
							</div>
						</div>
						<div class="row">
							<div class="col-5">
								<div>
									<img src='./resources/variability/car-truck/car_on_the_street.png' width="100%" style="padding-right: 1%;">
								</div>
							</div>
							<div class="col-2">
								<div>
									<img src='resources/arrow.png' width="100%" style="padding-right: 1%; max-height: 200px;">
									<!-- <p style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 5vw;margin-top: 40px;" width="100%" >&#8594;</p> -->
								</div>
							</div>
							<div class="col-5">
								<div id="car-truck" class="carousel slide carousel-fade" data-bs-ride="carousel" data-bs-interval="1000" >
									<div class="carousel-inner">
										<div class="carousel-item active">
											<img class="d-block" width="100%" src="./resources/variability/car-truck/1.png">
										</div>
										<div class="carousel-item">
											<img class="d-block" width="100%" src="./resources/variability/car-truck/2.png">
										</div>
										<div class="carousel-item">
											<img class="d-block" width="100%" src="./resources/variability/car-truck/3.png">
										</div>
										<div class="carousel-item">
											<img class="d-block" width="100%" src="./resources/variability/car-truck/4.png">
										</div>
									</div>
								</div>
							</div>
						</div>
					</div>
				</div>
				<div class="col">
					<div class="container-fluid">
						<div class="row">
							<div class="col-5">
								<div class="">
									<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px; "><br>A <span style="color:#FF0000">cartoon</span> of a <span style="color:#FF0000">cat</span></h1>				
								</div>
							</div>
							<div class="col-2">
							</div>
							<div class="col-5">
								<div class="">
									<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px; "><br>An <span style="color:#FF0000">origami</span> of a <span style="color:#FF0000">dog </span></h1>				
								</div>
							</div>
						</div>
						<div class="row">
							<div class="col-5">
								<div>
									<img src='resources/comparisons/cat-bear/cartoon_0.jpg' width="100%" style="padding-right: 1%;">
								</div>
							</div>
							<div class="col-2">
								<div>
									<img src='resources/arrow.png' width="100%" style="padding-right: 1%; max-height: 200px;">
									<!-- <p style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 5vw;margin-top: 40px;" width="100%" >&#8594;</p> -->
								</div>
							</div>
							<div class="col-5">
								<div id="origami-dog" class="carousel slide carousel-fade" data-bs-ride="carousel" data-bs-interval="1000">
									<div class="carousel-inner">
										<div class="carousel-item active">
											<img class="d-block" width="100%" src="./resources/variability/origami-dog/1.png">
										</div>
										<div class="carousel-item">
											<img class="d-block" width="100%" src="./resources/variability/origami-dog/2.png">
										</div>
										<div class="carousel-item">
											<img class="d-block" width="100%" src="./resources/variability/origami-dog/3.png">
										</div>
										<div class="carousel-item">
											<img class="d-block" width="100%" src="./resources/variability/origami-dog/4.png">
										</div>
									</div>
								</div>
							</div>
						</div>
					</div>
				</div>
				<div class="col">
					<div class="container-fluid">
						<div class="row">
							<div class="col-5">
								<div class="">
									<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px;">A <span style="color:#FF0000">cartoon</span> of a <span style="color:#FF0000"><br>castle</span></h1>				
								</div>
							</div>
							<div class="col-2">
							</div>
							<div class="col-5">
								<div class="">
									<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px;">An <span style="color:#FF0000">embroidery</span> of a <span style="color:#FF0000"><br>temple</span></h1>				
								</div>
							</div>
						</div>
						<div class="row">
							<div class="col-5">
								<div>
									<img src='./resources/variability/embroy/cartoon_17.jpg' width="100%" style="padding-right: 1%;">
								</div>
							</div>
							<div class="col-2">
								<div>
									<img src='resources/arrow.png' width="100%" style="padding-right: 1%; max-height: 200px;">
									<!-- <p style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 5vw;margin-top: 40px;" width="100%" >&#8594;</p> -->
								</div>
							</div>
							<div class="col-5">
								<div id="car-truck" class="carousel slide carousel-fade" data-bs-ride="carousel" data-bs-interval="1000" >
									<div class="carousel-inner">
										<div class="carousel-item active">
											<img class="d-block" width="100%" src="./resources/variability/embroy/1.png">
										</div>
										<div class="carousel-item">
											<img class="d-block" width="100%" src="./resources/variability/embroy/2.png">
										</div>
										<div class="carousel-item">
											<img class="d-block" width="100%" src="./resources/variability/embroy/3.png">
										</div>
										<div class="carousel-item">
											<img class="d-block" width="100%" src="./resources/variability/embroy/4.png">
										</div>
									</div>
								</div>
							</div>
						</div>
					</div>
				</div>
				<div class="col">
					<div class="container-fluid">
						<div class="row">
							<div class="col-5">
								<div class="">
									<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px; ">A <span style="color:#FF0000">painting</span> of a <span style="color:#FF0000"><br>goldfish</span></h1>				
								</div>
							</div>
							<div class="col-2">
							</div>
							<div class="col-5">
								<div class="">
									<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px; ">A <span style="color:#FF0000">video-game</span> of a <span style="color:#FF0000"><br>shark</span></h1>				
								</div>
							</div>
						</div>
						<div class="row">
							<div class="col-5">
								<div>
									<img src='resources/variability/goldfish-shark/painting_27.jpg' width="100%" style="padding-right: 1%;">
								</div>
							</div>
							<div class="col-2">
								<div>
									<img src='resources/arrow.png' width="100%" style="padding-right: 1%; max-height: 200px;">
									<!-- <p style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 5vw;margin-top: 40px;" width="100%" >&#8594;</p> -->
								</div>
							</div>
							<div class="col-5">
								<div id="origami-dog" class="carousel slide carousel-fade" data-bs-ride="carousel" data-bs-interval="1000">
									<div class="carousel-inner">
										<div class="carousel-item active">
											<img class="d-block" width="100%" src="./resources/variability/goldfish-shark/1.png">
										</div>
										<div class="carousel-item">
											<img class="d-block" width="100%" src="./resources/variability/goldfish-shark/2.png">
										</div>
										<div class="carousel-item">
											<img class="d-block" width="100%" src="./resources/variability/goldfish-shark/3.png">
										</div>
										<div class="carousel-item">
											<img class="d-block" width="100%" src="./resources/variability/goldfish-shark/4.png">
										</div>
									</div>
								</div>
							</div>
						</div>
					</div>
				</div>
			</div>

<!-- 
			<div class="row" style="margin-left: 6%">
				<div class="column" style="width: 22% !important;">
					<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px;">A photo of a <span style="color:#FF0000">car</span> on the side of the street</h1>				
				</div>
				<div class="column" style="width: 22% !important; margin-left:6%">
					<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px;">A photo of a <span style="color:#FF0000">truck</span> on the side of the street</h1>				
				</div>
				<div class="column" style="width: 22% !important;">
					<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px;">A <span style="color:#FF0000">cartoon</span> of a <span style="color:#FF0000">cat</span></h1>				
				</div>
				<div class="column" style="width: 22% !important;  margin-left:6%">
					<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px;">An <span style="color:#FF0000">origami</span> of a <span style="color:#FF0000">dog </span></h1>				
				</div>
			</div> -->
			<!-- <div class="row" style="margin-left: 6%;">
				<div class="column" style="width: 22% !important; margin-right:6%">
					<img id="car-truck1" class="round" width="100%" src='./resources/variability/car-truck/car_on_the_street.png'>
				</div>
                <div class="column" style="width: 22% !important;">
					<div id="car-truck" class="carousel slide carousel-fade" data-bs-ride="carousel" data-bs-interval="1000">
						<div class="carousel-inner">
						<div class="carousel-item active">
							<img class="d-block" width="100%" src="./resources/variability/car-truck/1.png">
						</div>
						<div class="carousel-item">
							<img class="d-block" width="100%" src="./resources/variability/car-truck/2.png">
						</div>
						<div class="carousel-item">
							<img class="d-block" width="100%" src="./resources/variability/car-truck/3.png">
						</div>
                        <div class="carousel-item">
							<img class="d-block" width="100%" src="./resources/variability/car-truck/4.png">
						</div>
						</div>
					</div>
				</div> -->
                <!-- <div class="column" style="width: 22% !important; margin-right:6%">
					<img id="origami-dog1" class="round" width="100%" src='./resources/variability/origami-dog/original.png'>
				</div>
                <div class="column" style="width: 22% !important;">
					<div id="origami-dog" class="carousel slide carousel-fade" data-bs-ride="carousel" data-bs-interval="1000">
						<div class="carousel-inner">
						<div class="carousel-item active">
							<img class="d-block" width="100%" src="./resources/variability/origami-dog/1.png">
						</div>
						<div class="carousel-item">
							<img class="d-block" width="100%" src="./resources/variability/origami-dog/2.png">
						</div>
						<div class="carousel-item">
							<img class="d-block" width="100%" src="./resources/variability/origami-dog/3.png">
						</div>
                        <div class="carousel-item">
							<img class="d-block" width="100%" src="./resources/variability/origami-dog/4.png">
						</div>
						</div>
					</div>
				</div>

			</div> -->

			<!-- <div class="row" style="margin-left: 6%">
				<div class="column" style="width: 22% !important;">
					<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px;">A <span style="color:#FF0000">cartoon</span> of a <span style="color:#FF0000">castle</span></h1>				
				</div>
				<div class="column" style="width: 22% !important; margin-left:6%">
					<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px;">An <span style="color:#FF0000">embroidery</span> of a <span style="color:#FF0000">temple</span></h1>				
				</div>
				<div class="column" style="width: 22% !important;">
					<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px;">A <span style="color:#FF0000">painting</span> of a <span style="color:#FF0000">goldfish</span></h1>				
				</div>
				<div class="column" style="width: 22% !important;  margin-left:6%">
					<h1 style="font-weight: normal; font-family: 'Comic Sans MS'; font-size: 16px;">A <span style="color:#FF0000">video-game</span> of a <span style="color:#FF0000">shark</span></h1>				
				</div>
			</div> -->
            <!-- <div class="row" style="margin-left: 6%;">
				<div class="column" style="width: 22% !important; margin-right:6%">
					<img id="cartoon1" class="round" width="100%" src='./resources/variability/embroy/cartoon_17.jpg'>
				</div>
                <div class="column" style="width: 22% !important;">
					<div id="cartoon" class="carousel slide carousel-fade" data-bs-ride="carousel" data-bs-interval="1000">
						<div class="carousel-inner">
						<div class="carousel-item active">
							<img class="d-block" width="100%" src="./resources/variability/embroy/1.png">
						</div>
						<div class="carousel-item">
							<img class="d-block" width="100%" src="./resources/variability/embroy/2.png">
						</div>
						<div class="carousel-item">
							<img class="d-block" width="100%" src="./resources/variability/embroy/3.png">
						</div>
                        <div class="carousel-item">
							<img class="d-block" width="100%" src="./resources/variability/embroy/4.png">
						</div>
						</div>
					</div>
				</div>
                <div class="column" style="width: 22% !important; margin-right:6%">
					<img id="goldfish-shark1" class="round" width="100%" src='./resources/variability/goldfish-shark/painting_27.jpg'>
				</div>
                <div class="column" style="width: 22% !important;">
					<div id="goldfish-shark" class="carousel slide carousel-fade" data-bs-ride="carousel" data-bs-interval="1000">
						<div class="carousel-inner">
						<div class="carousel-item active">
							<img class="d-block" width="100%" src="./resources/variability/goldfish-shark/1.png">
						</div>
						<div class="carousel-item">
							<img class="d-block" width="100%" src="./resources/variability/goldfish-shark/2.png">
						</div>
						<div class="carousel-item">
							<img class="d-block" width="100%" src="./resources/variability/goldfish-shark/3.png">
						</div>
                        <div class="carousel-item">
							<img class="d-block" width="100%" src="./resources/variability/goldfish-shark/4.png">
						</div>
						</div>
					</div>
				</div>

			</div> -->
			<!-- starry night
			<div class="row" style="margin-left: 15%;">
				<div class="column" style="width: 20% !important;">
					<img id="starry_night_roi" class="round" width="100%" style="margin-top:6%" src="./resources/data/starry_night.png">
				</div>
				<div class="column" style="width: 33% !important;">
					<img id="starry_night_roi_patches" class="round" border="1px solid" width="100%" src="./resources/ROI/starry_night/starry_night_patches.png">
				</div>
				<div class="column" style="width: 33% !important;">
					<div id="carousel_starry_night_roi" class="carousel slide carousel-fade" data-bs-ride="carousel">
						<div class="carousel-inner">
						<div class="carousel-item">
							<img class="d-block" width="100%" src="./resources/ROI/starry_night/1.png">
						</div>
						<div class="carousel-item active">
							<img class="d-block" width="100%" src="./resources/ROI/starry_night/2.png">
						</div>
						<div class="carousel-item">
							<img class="d-block" width="100%" src="./resources/ROI/starry_night/3.png">
						</div>
						</div>
					</div>
				</div>
			</div> -->

		</center>

		<br>
		<hr id="paper">

		<!-- paper -->
		<table align=center class="table"  style="max-width: 750px">
			<center><h1 style="margin-top: 65px;">Paper</h1></center>
			<tr>
				<td>
					<a href='resources/DDPMinversion_paper.pdf' target="_blank" rel="noopener noreferrer">
						<img alt="" class="layered-paper-big" style="height:100%; max-height: 175px;" src="./resources/paper.png"/>
					</a></td>
				<td><span style="font-size:14pt"><b>An Edit Friendly DDPM Noise Space: <br> Inversion and Manipulations.</b><br>
					Inbar Huberman-Spiegelglas, Vladimir Kulikov, Tomer Michaeli.</span>
					<br>
					<span style="font-size:14pt"><a href='https://arxiv.org/abs/2304.06140' target="_blank" rel="noopener noreferrer">[Arxiv]</a></span>
					<span style="font-size:14pt"><a href='resources/DDPMinversion_paper.pdf' target="_blank" rel="noopener noreferrer">[Paper]</a></span>
					</span>
				</td>
			</tr>
		</table>
		<p class="section">&nbsp;</p>
		<p class="section" id="bibtex"><b>Bibtex</b></p>
		<table border="0">
		<tbody>
			<pre class="command-copy" 
			style=" display: block;
					background: #eee;
					white-space: pre;
					-webkit-overflow-scrolling: touch;
					max-width: 100%;
					min-width: 100px;
					border-radius: 20px;
					padding: 0;;">

			@article{HubermanSpiegelglas2023,
				title      = {An Edit Friendly DDPM Noise Space: Inversion and Manipulations},
				author     = {Huberman-Spiegelglas, Inbar and Kulikov, Vladimir and Michaeli, Tomer},
				journal    = {arXiv preprint arXiv:2304.06140},
				year       = {2023}
			}
			</pre>
		</tbody>
		</table>
		<!-- <center>
			<table class="table" width="1">
				<tbody>
				<tr>
					<td style="font-size:14pt; text-align: left; padding-right: 40px !important;">
						Our official code implementation will be uploaded soon</a>.

					</td>
				</tr>
				</tbody>
			</table>
		</center> -->
		<!-- <center>
			<table class="table" width="1" style="max-width: 400px;">
				<tbody>
					<tr>
						<td width="50%" style="text-align: center;">
							<a target="_blank" rel="noopener noreferrer" href="resources/sinddm_supp.pdf" style="color:black !important;">
								<img alt="" id="supp" class="round" width="100%" style="max-width:150px" border="1px solid" src="./resources/supp.png"/>
							</a>
						</td>
						<td width="50%" style="text-align: center;">
							<a href="https://github.com/fallenshock/SinDDM" target="_blank" rel="noopener noreferrer">
								<img alt="" id="github_logo" class="round" width="100%" style="max-width:150px" src="./resources/github_logo.png"/>
							</a>
						</td>

					</tr>
					<tr>
						<td style="font-size:14pt; text-align: center;">
							<a href='resources/sinddm_supp.pdf' target="_blank" rel="noopener noreferrer">[Supplementary]</a>
						</td>
						<td style="font-size:14pt; text-align: center;">
							<a target="_blank" rel="noopener noreferrer" href='https://github.com/fallenshock/SinDDM'>[Code]</a>
						</td>

					</tr>
				</tbody>
			</table> -->

		<!-- </center> -->

		<hr>
		<br>

		<!-- acknowledgement
		<table align=center class="table">
			<tr>
				<td>
					<left>
						<center><h2>Acknowledgements</h2></center>
						This webpage is inspired by the template that was originally made by <a target="_blank" rel="noopener noreferrer" href="http://web.mit.edu/phillipi/">Phillip Isola</a> and
						<a target="_blank" rel="noopener noreferrer" href="http://richzhang.github.io/">Richard Zhang</a> for a <a target="_blank" rel="noopener noreferrer" href="http://richzhang.github.io/colorization/">colorful</a> ECCV project;
						the code for the original template can be found <a target="_blank" rel="noopener noreferrer" href="https://github.com/richzhang/webpage-template">here</a>.<br>
						We thank <a target="_blank" rel="noopener noreferrer" href="https://www.linkedin.com/in/hilamanor/">Hila Manor</a> for the help with this website.<br>
						A lot of features are taken from <a target="_blank" rel="noopener noreferrer" href="https://getbootstrap.com/">bootstrap</a>. All icons are taken from <a target="_blank" rel="noopener noreferrer" href="https://fontawesome.com/">font awesome.</a>
					</left>
				</td>
			</tr>
		</table> -->

	<br>

</div>

<!-- js scripts -->
<script>
	$(document).ready(function () {
		$('#myDropdown .dropdown-menu').on({
	"click":function(e){
      e.stopPropagation();
    }
});
$('.closer').on('click', function () {
    $('.btn-group').removeClass('open');
});

});

// sticky navbar 
document.addEventListener("DOMContentLoaded", function(){
  window.addEventListener('scroll', function() {
      if (window.scrollY > 50) {
        document.getElementById('navbar_top').classList.add('fixed-top');
        // add padding top to show content behind navbar
        navbar_height = document.querySelector('.navbar').offsetHeight;
        document.body.style.paddingTop = navbar_height + 'px';
      } else {
        document.getElementById('navbar_top').classList.remove('fixed-top');
         // remove padding top from body
        document.body.style.paddingTop = '0';
      } 
  });
}); 

// up button
//Get the button
let mybutton = document.getElementById("btn-back-to-top");

// When the user scrolls down 20px from the top of the document, show the button
window.onscroll = function () {
  scrollFunction();
};

function scrollFunction() {
  if (
    document.body.scrollTop > 20 ||
    document.documentElement.scrollTop > 20
  ) {
    mybutton.style.display = "block";
  } else {
    mybutton.style.display = "none";
  }
}
// When the user clicks on the button, scroll to the top of the document
mybutton.addEventListener("click", backToTop);

function backToTop() {
  document.body.scrollTop = 0;
  document.documentElement.scrollTop = 0;
}

</script>

</body>
</html>
