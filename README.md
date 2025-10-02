# Skull2Face

AI for reconstructing plausible faces from skulls. 

This repository provides a toy implementation of the concept described here: (https://www.observedimpulse.com/2025/09/skull-to-face-using-ai-to-recreate-lost_9.html?m=1).

The long-term vision is to develop machine learning models that take 3D skull geometry as input and generate plausible face reconstructions while explicitly representing uncertainty. This can support research in paleoanthropology, forensic science, and evolutionary biology.

Right now, this repo includes a **simplified PyTorch demo** that illustrates the basic pipeline structure. Here is a fuller description:


Skull 2 Face: Using AI to Recreate the Lost Faces of Human Evolution
  

Teaching AI to See the Faces in Fossil Skulls

What did our extinct prehistoric ancestors look like? For more than a century, artists have tried to rebuild ancient human faces from skulls. They were forced to use their imagination and ingenuity to do it. Until recently, that meant the faces of Neanderthals, Australopithecines, Homo floresiensis, and earlier hominins were partly unanswerable questions. Modern AI changes the terms of that question.

The idea is simple. We can show an AI model thousands of paired examples where the input is a 3-D skull and the output is the real face that belonged to it. We start with humans (where matched medical scans exist), then add living apes and monkeys to teach the system how primate skull shape influences the soft tissue above it. Once the model learns those relationships, we can point it at fossil skulls and ask: “Given this bone structure, what faces are most consistent with what you’ve learned?” Even though cartilage and soft tissues don’t fossilize, the skull gives strong hints. Its bony architecture sharply constrains the overall head shape, including, forehead slope, brow, zygomatic arches, dental arcade, mandible, nasal aperture, chin, and muscle attachment sites.



If we can feed enough of these skull/face pairs to a machine learning model it will starts to learn the regularities and turn them into a mathematical function. It will learn, for example, how the curve of the zygomatic arch relates to cheek contour, how the nasal aperture and spine relate to the bridge of the nose, and how jaw robustness relates to mouth shape. Widening the training set with non-human primates will keep the model from overfitting to modern. The apes and monkeys will also extend the range of skull shapes and soft tissues and provide a phylogenetic bridge so that all the model must do is create interpolations. This project should use a large, highly diverse sample of humans as well as several member from all of the great apes (chimps, bonobos, gorillas, orangutans), lesser apes (gibbons), and possibly also monkeys as well. It might help to remove the hair from at least some of the primate 3D models so that the features that are normally covered by fur can be mapped.

This system would effectively be teaching the AI how to translate between skulls and faces (and remember translation is what the “transformer” AI architecture was originally developed to accomplish). So, the AI would learn a skull-to-face “vocabulary” from living species, and then use it to translate fossil skulls. It’s like a police sketch process, except now the skull is the witness.

There are many human artists who have created paleoart depictions of prehistoric human ancestors. Some of these paintings are exquisite and highly evocative. The human artists make their design decisions using the extremely sophisticated neural networks in their heads, but the job would be better accomplished by machines due to their computational advantages. Making a precise statistical prediction about a face from a reference skull involves attention to more features and patterns than humans are capable of keeping in mind at once. There are all kinds of hidden, latent correlations that the human mind does not notice, but that machine learning will pick up on very easily. This is why we should use AI.




How Would This System Work?

At its core the machine learning system has two halves. The first part of this AI system is a "transformer" which is a type of neural network that pays attention to the geometric relationships in the 3D skull, how different parts of the bone relate to each other and how they shape the face. It would work as a geometric encoder that studies the skull as a true 3-D object rather than a photograph. We would feed it cleaned skull meshes (or dense point clouds) in a standard pose. This 3-D transformer would learn which bony regions co-vary, distilling the skull into a compact, numerical description.

This first system would then lead to an image generation system that would create a prediction of the face it expects to lie on top of each skull. Thus, during training, it would try to generate (using an imagery generation technique called diffusion) a face for each skull and it would get feedback based on the actual (ground truth) face. Over time, it would learn the “mapping” of skulls to face and get really good at understanding which skull features correspond to which facial features.

The best fit here for the face generator may be a “conditional diffusion model” that would take the transfomer’s latent description of the skull and sample many faces conditioned on it. Most state-of-the-art image generators today are from the “diffusion family.” They dominate because they’re stable to train, scale well, condition cleanly on text or other inputs, and naturally produce diverse samples rather than collapsing to a single look. Adobe Firefly, Midjourney, DALL·E 3, Stable diffusion, Google’s Imagen line, Runway’s recent models, and many research systems are all diffusion/flow variants under the hood. For a single, simple stack, a diffusion network over a mesh/point representation with the skull encoder’s tokens plugged in via cross-attention is probably the clean, modern choice.

Hundreds of pairs might get this model off the ground, but it would need thousands of human and primate pairs to show some precision. Of course, we wouldn’t want to reinvent the wheel. So, we wouldn’t start entirely from scratch. Instead, we would use something known as transfer learning, where we start with an existing model that’s already been trained on a large amount of 3D shape or image data. These models have a good baseline understanding of physical general patterns, even those related to faces. Then we would fine-tune it with the skull-face pairs so it learns the specialized task of transforming skulls to faces. By leveraging an existing image generation or 3D model and then adapting it, we would save a lot of time and data.

Once that mapping is learned, we can turn it toward the past. We can present a well-preserved Neanderthal cranium or the small skull of Homo floresiensis and ask the model to propose faces that are consistent with the bone. Some of the skulls coming from paleoanthropology departments are crushed. However, we already have algorithms that can digitally “uncollapse” a fossil that was crushed in the ground so that the AI model sees a truer shape before it begins to infer a face. The right output is not a single portrait. Bone to face is an underdetermined problem and many faces can fit the same skull. A responsible system would produce an ensemble of possibilities that agree where the skull is informative and vary where it is silent. The bony frame nails down head shape, brow, jawline and overall proportions. The details that live in cartilage and fat, such as the tip of the nose, the lips and the ears, carry wider uncertainty. A good system would make that uncertainty visible and apparent to users.

Next, we would want to show that transforming skulls to faces works on humans with measurable accuracy and well calibrated uncertainty. Then we would want to show that this generalizes to other primates by holding out a species (such as orangutans) from training, but then testing the model’s ability to recreate that species face after training. Only then would we present fossils. With a team of researchers, or with a sufficiently advanced AI agent, the fossil could be presented alongside expert commentary and comparisons to traditional hand-built reconstructions.







What Would This System Give Us?

Several extra sources of information can be used to help reduce guesswork. Forensic science has measured average soft tissue thickness at standard points on the face across age, sex and ancestry. Those tables can be used as anchors. Ancient DNA, where we have it, could inform pigmentation and other features. Known information pertaining to the genes that influence skin, hair and eye color can narrow the palette for Neanderthals and some early modern humans. Comparative anatomy helps too. A specimen placed on a particular branch of the family tree should look like its neighbors on that branch. A reconstruction of Australopithecus should not drift toward a modern human nose, and a reconstruction of early Homo should not drift toward a chimp.

A museum could present the results in a way that reveals both the power and the limits of the method. Imagine standing in front of a fossil cranium and a rotating 3-D viewer. You can toggle through twenty plausible faces derived from the same skull. A simple overlay highlights the regions where the model is most confident in green and least confident in red. A control lets you alter the features based on ancient DNA when it exists. It invites the public to see how bone constrains flesh and where knowledge gives way to uncertainty. Reconstructions should be clearly labeled as probabilistic and training data should be de-identified.

What might we actually see if we were to build this model? Neanderthals provide a best case because we have several complete crania and high-quality DNA. The ensemble would likely be tight where bone speaks strongly and more variable around the nose and lips. The tiny LB1 skull from Flores would show a different pattern, with robust jaws and a short midface but wider uncertainty in soft tissue. A classic Australopithecus like “Mrs. Ples” would land in between ape-like prognathism and human-like flattening. Denisovans are the opposite case. For Denisova hominins, we have remarkable DNA and finger fossils, but no fossil skull so this technique could not be applied to them (yet).

This project also belongs to a larger idea that modern AI can act as telescope capable of peering back through time. Accordingly, this technique could be used for dinosaurs or any extinct creatures. We could recreate a T. Rex face based on a system trained on birds and reptiles. Furthermore, I have created a genome prediction pipeline that could estimate a plausible genome for when we don’t have a genome for the species in question.

https://www.observedimpulse.com/2025/07/ai-mediated-reconstruction-of-dinosaur.html

Conclusion

I remember back in my twenties, I thought the face of an extinct hominin was one of the great mysteries of science and something I very much longed to see. Although these faces will never be known with finality, with careful methods, we can recover their outlines and likely expressions. We can give museums new tools to teach. We can give readers a stronger sense of kinship. The result is not just a new picture in a textbook. It is a new way to talk with the public about evidence. A single confident portrait invites argument about smiles and haircuts. An interactive ensemble would invite questions about how bone shapes flesh, about what DNA can and cannot tell us, and about how evolution channels variation. It lets us look an ancestor in the eyes while remembering that some parts of that gaze come from rock and measurement, and some parts come from honest uncertainty.

I have always thought it unfortunate that most modern depictions make hominins look brutish. I think the main take away from high fidelity facial recreations is that we will see these people were beautiful and noble. Some of them would’ve been diminutive and adorable. Others would have been intimidating due to their size and robusticity. Some of the phylogenetically older ones might be eerie looking, but their visages would help us see our continuity with the rest of the primate order. But I predict here that we will see the intellect in their eyes, we will find them attractive, and we will see them as cousins and equals. I believe that they will invariably look interesting, making us want to reach out, talk, and engage with them. And I think, in the not-so-distant future, AI could help us have a scientifically accurate, communicative interaction with the avatar of a homo erectus individual. Before we wrap here, let’s take a look at which hominin species we actually have skulls from.

Here’s a list of hominin taxa with substantially complete crania/skulls (good enough for 3-D modeling) that this pipeline could take as inputs. They are grouped here roughly by era, with exemplar specimens in brackets.

• Sahelanthropus tchadensis — Toumaï cranium [TM 266-01-060-1]
• Australopithecus anamensis — near-complete cranium [MRD-VP-1/1]
• Australopithecus afarensis — adult A.L. 444-2; juvenile “Selam” [DIK-1-1]
• Australopithecus africanus — “Mrs. Ples” [STS 5]; “Little Foot” [StW 573]
• Australopithecus sediba — MH1 (“Karabo”), MH2 crania
• Paranthropus aethiopicus — “Black Skull” [KNM-WT 17000]
• Paranthropus boisei — “Zinj” [OH 5], KNM-ER 406
• Paranthropus robustus — SK 48; DNH 155
• Kenyanthropus platyops — cranium KNM-WT 40000 (distorted but complete enough)
• Homo habilis — KNM-ER 1813; OH 24 “Twiggy” (reconstructed)
• Homo rudolfensis — KNM-ER 1470
• Dmanisi early Homo (often H. georgicus / early H. erectus) — D2280, D2700, D4500
• Homo ergaster — KNM-ER 3733, 3883; Turkana Boy skull [KNM-WT 15000]
• Homo erectus — Sangiran 17; Zhoukoudian crania; Mojokerto; Ngandong
• Homo heidelbergensis sensu lato (incl. “H. rhodesiensis” / “H. bodoensis”)
• East Asian mid-Pleistocene archaic Homo — Dali, Jinniushan, Maba
• Homo longi (“Dragon Man”) — Harbin cranium (very complete)
• Homo naledi — several near-complete crania (e.g., “Neo”; composite but model-ready)
• Homo neanderthalensis — (e.g. La Chapelle-aux-Saints 1, La Ferrassie 1)
• Homo floresiensis — LB1 (nearly complete), LB6 (partial)
• Homo sapiens (anatomically modern) — Herto (BOU-VP-16/1), Omo 1, Skhul, Qafzeh

