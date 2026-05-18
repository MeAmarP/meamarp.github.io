// A $( document ).ready() block.
$( document ).ready(function() {

	// DropCap.js
	var dropcaps = document.querySelectorAll(".dropcap");
	window.Dropcap.layout(dropcaps, 2);

	// Responsive-Nav (only init if legacy .nav-collapse element exists)
	if (document.querySelector(".nav-collapse")) {
		var nav = responsiveNav(".nav-collapse");
	}

	// Round Reading Time
    $(".time").text(function (index, value) {
      return Math.round(parseFloat(value));
    });

});

// Scroll-spy: update bottom-dock active tab as user scrolls (home page only)
(function () {
  if (!document.querySelector('.portfolio-home')) return;

  var tabs = document.querySelectorAll('.top-dock .dock-tab');

  // Listed bottom-to-top: first match wins when section top is above viewport midpoint
  var sectionTabMap = [
    { id: 'contact',  href: '/#contact' },
    { id: 'blogs',    href: '/#blogs' },
    { id: 'projects', href: '/#projects' },
  ];

  function updateActive() {
    var threshold = window.innerHeight * 0.75;
    var activeHref = '/#home';
    for (var i = 0; i < sectionTabMap.length; i++) {
      var el = document.getElementById(sectionTabMap[i].id);
      if (el && el.getBoundingClientRect().top <= threshold) {
        activeHref = sectionTabMap[i].href;
        break;
      }
    }
    tabs.forEach(function (tab) {
      tab.classList.toggle('active', tab.getAttribute('href') === activeHref);
    });
  }

  window.addEventListener('scroll', updateActive, { passive: true });
  updateActive();
}());


