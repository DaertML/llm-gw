# llm-gw
Repo to host production/dev environments for enterprise LLM systems

# Introduction
DaertML since its inception has been pretty much a research lab, and that is great because we are at the forefront of the AI ecosystem, but at the same time, we are lacking to cover the needs of enterprises and end users. Because of that, some of the recent repos try to cover this gap, and become more enterprise aware.

In this case, the idea is to provide increasingly more "production ready" environments that you can run to cover the different needs that an enterprise has with respect to AI. The plan is to not use paid AI providers, and cover all the needs with local infrastructure, that is open and offered to the end user to easily consume and extend for their own use case.

# Influence
If one thinks about AI in terms of a swarm of connected systems, one gets to the metaphor of how Factorio works: there are plenty of services that relate to each other, that take the byproduct of each of them and processes them further and makes new byproducts. In that sense, implementing all the AI needs as microservices that you link together may feel like the best choice, keeping the AI gateway in the middle to communicate and orchestrate the needed services. In this manner, as needs arise, one can choose to grow different parts separately, remove them, or add them later on as needed.

In a similar sense, if you look at cloud/systems designs in which there are plenty of boxes and relationships between services... forget about that, that's the final picture, you should try to build he picture from scratch, by making incremental changes, as usually is how it is done to match the real needs and not to oversell.

# Developed environments
- simple: simplest environment that can be created
- simple-tracking: added a postgresql DB to keep track of the consumption
- simple-tracking-web: adds the web UI of litellm and the Open Web UI for LLM consumption
- simple-tracking-web-rag: adds a service that ingests, chunks and indexes documents, and wraps LLM calls around it, so you can communicate with it through the Open Web UI

# Special thanks
I would like to thank the effort of Pablo Dafonte Iglesias, again, as someone who has guided me into trying this, so that, we go to the next step in covering AI needs. That's what I'm missing the enterprise part of thinking :)
