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


