$(document).ready(function(){

     $(document).on("change", "#make", function () {
       var make = $('#make').val();
       $('#model').find('option').remove().end()
         $.ajax({
              type : 'POST',
              url : '/selectCarModel',
              contentType: 'application/json;',
              data :JSON.stringify({
                   make:make
                  }),
               success: function (result) {
                  JSON.parse(result, function (index, value) {
                      if(index == "model"){
                           $('#model').append($('<option/>', {
                            value: value,
                            text : value
                        }));
                      }

                  });

                  $('#model_year').find('option').remove().end()
                   $.ajax({
                      type : 'POST',
                      url : '/selectModelYear',
                      contentType: 'application/json;',
                      data :JSON.stringify({
                           make:make
                          }),
                       success: function (result) {
                       JSON.parse(result, function (index, value) {
                              if(index == "model_year") {
                                  $('#model_year').append($('<option/>', {
                                      value: value,
                                      text: value
                                  }));
                              }
                          });
                       }
                 });
               }
         });
     })

     $(document).on("click", "#predictBtn", function () {

     var make = $('#make').val();
     var model = $('#model').val();
     var model_year = $('#model_year').val();
     var transmission = $('#transmission').val();
     var fuel_type = $('#fuel_type').val();
     var engine_capacity = $('#engine_capacity').val();
     var mileage = $('#mileage').val();


     if(engine_capacity && mileage && make!='- Select Make -' && model!='- Select Model -'){
       $.ajax({
          type : 'POST',
          url : '/predict',
          contentType: 'application/json;',
          data :JSON.stringify({
           make:make,
           model:model,
           model_year:model_year,
           transmission: transmission,
           fuel_type:fuel_type,
           engine_capacity:engine_capacity,
           mileage:mileage
          }),
           success: function (data) {
               console.log(data.prediction_text[0])
               var price = numberWithCommas(Math.round(data.prediction_text[0]))
               $("#predictedTxt").text("Rs."+price+".00")
            }
        });
      }else{
        alert("Please fill the empty fields")
      }
    });
});

function numberWithCommas(price) {
  return price.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}
