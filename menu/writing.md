---
layout: default
title: Writing
permalink: /writing/
---
<div class="portfolio-home">
	<section class="cards-section section">
		<div class="section-heading">
			<p class="eyebrow">All posts</p>
			<h2>Writing</h2>
		</div>
		<div class="card-grid two-wide">
			{%- for post in site.posts -%}
			<article class="content-card blog-card">
				<span class="card-kicker">{{ post.date | date: '%b %Y' }}</span>
				<h3>
					<a class="card-link" href="{{ post.url | relative_url }}">{{ post.title }}</a>
				</h3>
				<p>
					{%- if post.description -%}
						{{ post.description }}
					{%- else -%}
						{{ post.excerpt | strip_html | truncatewords: 28 }}
					{%- endif -%}
				</p>
			</article>
			{%- endfor -%}
		</div>
	</section>
</div>
